"""E9: combining finished models (D25). Both variants are chosen on validation only and written
as normal runs, so evaluate.py scores them like any other experiment.

  e9_ensemble    average of several arms' class probabilities (each arm's seed mean); members
                 picked by greedy forward selection on target_val macro AUC (at most 4; a member
                 is added only if it raises target_val by >= 0.002). Deterministic: s0 only.
  e9_fusion_lr   logistic regression on [Derm Foundation embedding || E1 EfficientNet-B0 embedding]
                 of the same seed, trained on source_train, C picked on source_val (seeds 0, 1, 2)

    python -m skincancer.combine
    python -m skincancer.combine --final ddi_test dermacon_test dermacon_other pad_dark_test midas_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import runs
from .config import load_config
from .foundation import LR_C, load_embeddings, write_run, xy
from .metrics import macro_auc

CANDIDATES = ["e1_effb0", "e1_r50", "e1_effb0_320", "e1_effb0_d7", "e1_fm_lr", "e1_fm_mlp", "e5_fm_lr",
              "e3_cc", "e3_clahe", "e3_crop", "e3_all", "e4_indian_loss", "e5s_f100", "e6_fusion",
              "e6_embed_mlp", "e6_texture_lr", "e7_coral_l100.0", "e7_coral_l1000.0", "e7_coral_l10000.0",
              "e8_r50"]
MAX_MEMBERS, MIN_GAIN = 4, 0.002


def arm_probs(exp: str, split: str, classes: list[str], runs_dir: str = "runs") -> pd.DataFrame:
    """Seed-mean class probabilities of one finished arm, indexed by image_id."""
    cols = [f"p_{c}" for c in classes]
    frames = [pd.read_csv(f).set_index("image_id")[cols] for f in sorted(Path(runs_dir, exp).glob(f"s*/preds/{split}.csv"))]
    if not frames:
        raise FileNotFoundError(f"{exp}: no predictions for {split}")
    ids = frames[0].index
    return pd.DataFrame(np.mean([f.loc[ids].to_numpy() for f in frames], axis=0), index=ids, columns=cols)


def greedy_select(P: dict[str, np.ndarray], y: np.ndarray) -> tuple[list[str], list[float]]:
    chosen, path, best = [], [], -np.inf
    while len(chosen) < MAX_MEMBERS:
        scores = {e: macro_auc(y, np.mean([P[m] for m in chosen + [e]], axis=0)) for e in P if e not in chosen}
        e, s = max(scores.items(), key=lambda kv: kv[1])
        if chosen and s < best + MIN_GAIN:
            break
        chosen.append(e)
        path.append(round(float(s), 4))
        best = s
    return chosen, path


def ensemble(cfg, manifest, classes, splits, final, force):
    label_col = cfg["data"]["label_col"]
    idx = {c: i for i, c in enumerate(classes)}
    tv = manifest[manifest["split"] == "target_val"]
    y = tv[label_col].map(idx).fillna(-1).astype(int).to_numpy()
    P = {e: arm_probs(e, "target_val", classes).loc[tv["image_id"]].to_numpy() for e in CANDIDATES}
    alone = {e: round(float(macro_auc(y, p)), 4) for e, p in P.items()}
    print("target_val macro AUC alone:", dict(sorted(alone.items(), key=lambda kv: -kv[1])))
    members, path = greedy_select(P, y)
    print("greedy members:", members, "target_val path:", path)
    data = {}
    for s in splits + final:
        df = manifest[manifest["split"] == s][["image_id"]].reset_index(drop=True)
        X = np.mean([arm_probs(e, s, classes).loc[df["image_id"]].to_numpy() for e in members], axis=0) if len(df) else np.zeros((0, len(classes)))
        data[s] = (df, X, None)
    write_run("e9_ensemble", 0, cfg, lambda X: X, data, splits, final, force,
              {"e9_members": members, "e9_target_val_path": path})


def fusion(cfg, manifest, classes, splits, final, force, seeds):
    label_col = cfg["data"]["label_col"]
    E = load_embeddings(cfg["features_dir"])
    need = list(dict.fromkeys(["source_train", "source_val"] + splits + final))
    for seed in seeds:
        data = {}
        for s in need:
            df, F, y = xy(s, E, manifest, label_col, classes)
            z = np.load(Path(cfg["runs_dir"]) / "e1_effb0" / f"s{seed}" / "emb" / f"{s}.npz", allow_pickle=True)
            C = pd.DataFrame(z["features"], index=z["image_id"]).loc[df["image_id"]].to_numpy()
            data[s] = (df, np.hstack([F, C]), y)
        _, Xtr, ytr = data["source_train"]
        _, Xva, yva = data["source_val"]
        scaler = StandardScaler().fit(Xtr)
        m = ytr >= 0
        Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
        fits = {c: LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(Ztr[m], ytr[m]) for c in LR_C}
        val = {c: macro_auc(yva, f.predict_proba(Zva)) for c, f in fits.items()}
        best = max(val, key=val.get)
        print(f"fusion s{seed} source_val macro AUC:", {c: round(v, 4) for c, v in val.items()}, "-> C =", best)
        clf = fits[best]
        write_run("e9_fusion_lr", seed, cfg, lambda X, clf=clf: clf.predict_proba(scaler.transform(X)), data,
                  splits, final, force, {"fm_head": "logistic", "fm_C": best, "e9_inputs": "derm_foundation+e1_effb0"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["source_val", "source_test", "target_val", "target_test", "ak_side"])
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["ensemble", "fusion"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["manifest"], low_memory=False)
    classes = runs.classes_of(cfg)
    if args.only != "fusion":
        ensemble(cfg, manifest, classes, args.splits, args.final, args.force)
    if args.only != "ensemble":
        fusion(cfg, manifest, classes, args.splits, args.final, args.force, args.seeds)


if __name__ == "__main__":
    main()
