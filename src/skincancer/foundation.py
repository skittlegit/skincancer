"""E1-FM: source-only baseline on frozen Derm Foundation embeddings (D15).

Heads trained on source_train, selected on source_val only, written as normal runs so
evaluate.py scores them like any other experiment:
  e1_fm_mlp   the E6 MLP head (seeds 0, 1, 2); compare with e6_embed_mlp (same head, E1 embedding)
  e1_fm_lr    linear probe, class-balanced logistic regression, C picked on source_val (s0 only:
              deterministic)
  e5_fm_lr    (--with-target) the linear probe trained on source_train + target_train, C picked
              on target_val: E5's question (does same-atlas dark-skin data help?) on this backbone

Embeddings come from scripts/derm_foundation_embed.py (run in .venv-tf).
    python -m skincancer.foundation --seeds 0 1 2
    python -m skincancer.foundation --seeds 0 1 2 --final ddi_test dermacon_test dermacon_other pad_dark_test midas_test
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import predict, runs
from .config import load_config
from .fusion import train_mlp
from .metrics import macro_auc

LR_C = [1e-3, 1e-2, 1e-1, 1.0]
LR_C_TARGET = [1e-4, 1e-3, 1e-2, 1e-1]   # e1_fm_lr chose the grid edge (1e-3), so extend down


def load_embeddings(features_dir: str) -> pd.DataFrame:
    parts = sorted((Path(features_dir) / "derm_foundation").glob("part_*.npz"))
    if not parts:
        raise SystemExit("no Derm Foundation embeddings: run scripts/derm_foundation_embed.py in .venv-tf")
    ids, embs = [], []
    for f in parts:
        z = np.load(f, allow_pickle=True)
        ids.append(z["image_id"])
        embs.append(z["embedding"].astype(np.float32))
    E = pd.DataFrame(np.vstack(embs), index=np.concatenate(ids))
    return E[~E.index.duplicated()]


def xy(split, E, manifest, label_col, classes):
    df = manifest[manifest["split"] == split]
    df = df[df["image_id"].isin(E.index)]
    idx = {c: i for i, c in enumerate(classes)}
    y = df[label_col].map(idx).fillna(-1).astype(int).to_numpy()
    return df[["image_id"]].reset_index(drop=True), E.loc[df["image_id"]].to_numpy(), y


def write_run(name, seed, base_cfg, prob, data, eval_splits, final, force, extra):
    cfg = dict(copy.deepcopy(base_cfg), name=name, seed=seed, **extra)
    rdir = runs.run_dir(cfg)
    if final:
        predict.check_ledger(str(rdir.as_posix()), final, force)
    (rdir / "preds").mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, rdir / "config.yaml")
    k = len(runs.classes_of(cfg))
    runs.write_json({"classes": runs.classes_of(cfg), "effective_prior": [1.0 / k] * k}, rdir / "train_prior.json")
    for split in eval_splits + final:
        df, X, _ = data[split]
        if len(df):
            runs.prediction_frame(df, prob(X), runs.classes_of(cfg), cfg).to_csv(rdir / "preds" / f"{split}.csv",
                                                                                 index=False)
    if final:
        predict.append_ledger(str(rdir.as_posix()), [s for s in final if len(data[s][0])])
    print(f"{name} s{seed}: done")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["source_val", "source_test", "target_val", "target_test", "ak_side"])
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-target", action="store_true", help="e5_fm_lr only (see module docstring)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["manifest"])
    label_col = cfg["data"]["label_col"]
    classes = runs.classes_of(cfg)
    E = load_embeddings(cfg["features_dir"])
    need = list(dict.fromkeys(["source_train", "target_train"] + args.splits + args.final))
    data = {s: xy(s, E, manifest, label_col, classes) for s in need}
    missing = {s: int((manifest["split"] == s).sum()) - len(data[s][0]) for s in need}
    assert not any(missing.values()), f"images without embeddings: {missing}"

    if args.with_target:
        return with_target(cfg, data, args)

    _, Xtr, ytr = data["source_train"]
    _, Xva, yva = data["source_val"]
    scaler = StandardScaler().fit(Xtr)
    m = ytr >= 0
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)

    # Linear probe: C chosen on source_val macro AUC (D1).
    fits = {C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Ztr[m], ytr[m]) for C in LR_C}
    val = {C: macro_auc(yva, f.predict_proba(Zva)) for C, f in fits.items()}
    best_c = max(val, key=val.get)
    print("linear probe source_val macro AUC:", {c: round(v, 4) for c, v in val.items()}, "-> C =", best_c)
    clf = fits[best_c]
    write_run("e1_fm_lr", 0, cfg, lambda X: clf.predict_proba(scaler.transform(X)), data, args.splits,
              args.final, args.force, {"fm_head": "logistic", "fm_C": best_c})

    for seed in args.seeds:
        net = train_mlp(Ztr, ytr, Zva, yva, len(classes), seed)
        prob = lambda X, net=net: torch.softmax(net(torch.tensor(scaler.transform(X), dtype=torch.float32)),  # noqa: E731
                                                1).detach().numpy()
        write_run("e1_fm_mlp", seed, cfg, prob, data, args.splits, args.final, args.force, {"fm_head": "mlp"})


def with_target(cfg, data, args):
    Xtr = np.vstack([data["source_train"][1], data["target_train"][1]])
    ytr = np.concatenate([data["source_train"][2], data["target_train"][2]])
    _, Xva, yva = data["target_val"]
    scaler = StandardScaler().fit(Xtr)
    m = ytr >= 0
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    fits = {C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Ztr[m], ytr[m])
            for C in LR_C_TARGET}
    val = {C: macro_auc(yva, f.predict_proba(Zva)) for C, f in fits.items()}
    best_c = max(val, key=val.get)
    print("source+target probe target_val macro AUC:", {c: round(v, 4) for c, v in val.items()}, "-> C =", best_c)
    clf = fits[best_c]
    write_run("e5_fm_lr", 0, cfg, lambda X: clf.predict_proba(scaler.transform(X)), data, args.splits,
              args.final, args.force, {"fm_head": "logistic", "fm_C": best_c, "fm_train": "source+target"})


if __name__ == "__main__":
    main()
