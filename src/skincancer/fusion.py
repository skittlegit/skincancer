"""E6: texture + CNN fusion on cached features (D9).

Per seed, from the frozen E1 model of that seed, trains three heads on source_train
(early stop on source_val) and writes each as a normal run, so evaluate.py scores them:
  e6_fusion      MLP on [embedding ‖ texture]
  e6_embed_mlp   same MLP on the embedding alone (control)
  e6_texture_lr  logistic regression on texture alone (floor)

    python -m skincancer.fusion --source-exp e1_effb0 --seeds 0 1 2
    python -m skincancer.fusion --source-exp e1_effb0 --seeds 0 1 2 --final ddi_test dermacon_test
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import predict, runs
from .data.dataset import ImageLoader
from .features.texture import FEATURE_NAMES, texture_features
from .metrics import macro_auc
from .models import FusionMLP

TEXTURE_PREPROCESS = {"color_constancy": True, "clahe": True, "lesion_crop": True}
TRAIN_SPLITS = ["source_train", "source_val"]


def _texture_one(args):
    image_id, path, data_root, cache_dir = args
    img = ImageLoader(data_root, TEXTURE_PREPROCESS, cache_dir)(image_id, path)
    return texture_features(img)


def texture_table(manifest: pd.DataFrame, cfg: dict, splits: list[str], workers: int = 4) -> pd.DataFrame:
    """Texture features for every image in `splits`, cached in features_dir (seed-independent)."""
    path = Path(cfg["features_dir"]) / "texture.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["image_id", *FEATURE_NAMES])
    need = manifest[manifest["split"].isin(splits) & ~manifest["image_id"].isin(cached["image_id"])]
    if len(need):
        jobs = [(r.image_id, r.path, cfg["data_root"], cfg.get("cache_dir")) for r in need.itertuples()]
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                feats = list(ex.map(_texture_one, jobs, chunksize=16))
        else:
            feats = [_texture_one(j) for j in jobs]
        new = pd.DataFrame(np.vstack(feats), columns=FEATURE_NAMES)
        new.insert(0, "image_id", need["image_id"].to_numpy())
        cached = pd.concat([cached, new], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False)
    return cached.set_index("image_id")


def _xy(split, emb, tex, manifest, cfg, classes, kind):
    ids, F = emb[split]
    T = tex.loc[ids].to_numpy(np.float32)
    X = {"fusion": np.hstack([F, T]), "embed": F, "texture": T}[kind]
    labels = manifest.set_index("image_id").loc[ids, cfg["data"]["label_col"]]
    idx = {c: i for i, c in enumerate(classes)}
    y = labels.map(idx).fillna(-1).astype(int).to_numpy()
    return ids, X, y


def train_mlp(Xtr, ytr, Xva, yva, k: int, seed: int, epochs=200, patience=15) -> FusionMLP:
    torch.manual_seed(seed)
    m = ytr >= 0
    Xtr, ytr = torch.tensor(Xtr[m], dtype=torch.float32), torch.tensor(ytr[m])
    counts = torch.bincount(ytr, minlength=k).float()
    weights = counts.sum() / (k * counts.clamp(min=1))
    net = FusionMLP(Xtr.shape[1], k)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss(weight=weights)
    best, best_state, stale = -np.inf, None, 0
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            crit(net(Xtr[b]), ytr[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auc = macro_auc(yva, torch.softmax(net(Xva_t), 1).numpy())
        if auc > best:
            best, best_state, stale = auc, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:  # None only if validation AUC was NaN every epoch
        net.load_state_dict(best_state)
    return net.eval()


def run_seed(source_exp: str, seed: int, eval_splits: list[str], final: list[str], force: bool, workers: int,
             runs_dir: str = "runs"):
    sdir = Path(runs_dir) / source_exp / f"s{seed}"
    model, cfg = runs.load_run(sdir)
    manifest = pd.read_csv(cfg["manifest"])
    classes = runs.classes_of(cfg)
    all_splits = list(dict.fromkeys(TRAIN_SPLITS + eval_splits + final))
    tex = texture_table(manifest, cfg, all_splits, workers)
    emb = {s: predict.embeddings(sdir, s, model, cfg, manifest) for s in all_splits
           if (manifest["split"] == s).any()}

    for kind, name in (("fusion", "e6_fusion"), ("embed", "e6_embed_mlp"), ("texture", "e6_texture_lr")):
        out_cfg = dict(copy.deepcopy(cfg), name=name, seed=seed, e6_source=str(sdir.as_posix()))
        rdir = runs.run_dir(out_cfg)
        if final:
            predict.check_ledger(str(rdir.as_posix()), final, force)
        (rdir / "preds").mkdir(parents=True, exist_ok=True)
        runs.save_config(out_cfg, rdir / "config.yaml")
        k = len(classes)
        runs.write_json({"classes": classes, "effective_prior": [1.0 / k] * k}, rdir / "train_prior.json")

        _, Xtr, ytr = _xy("source_train", emb, tex, manifest, cfg, classes, kind)
        _, Xva, yva = _xy("source_val", emb, tex, manifest, cfg, classes, kind)
        scaler = StandardScaler().fit(Xtr)
        if kind == "texture":
            m = ytr >= 0
            clf = LogisticRegression(class_weight="balanced", max_iter=5000).fit(scaler.transform(Xtr[m]), ytr[m])
            prob = lambda X: clf.predict_proba(scaler.transform(X))  # noqa: E731
        else:
            net = train_mlp(scaler.transform(Xtr), ytr, scaler.transform(Xva), yva, k, seed)
            prob = lambda X: torch.softmax(net(torch.tensor(scaler.transform(X), dtype=torch.float32)), 1).detach().numpy()  # noqa: E731

        for split in eval_splits + final:
            if split not in emb:
                continue
            ids, X, _ = _xy(split, emb, tex, manifest, cfg, classes, kind)
            df = pd.DataFrame({"image_id": ids})
            runs.prediction_frame(df, prob(X), classes, out_cfg).to_csv(rdir / "preds" / f"{split}.csv", index=False)
        runs.write_json({"source_run": str(sdir), "n_train": int((ytr >= 0).sum())}, rdir / "summary.json")
        if final:
            predict.append_ledger(str(rdir.as_posix()), [s for s in final if s in emb])
        print(f"{name} s{seed}: done")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-exp", default="e1_effb0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=["source_val", "source_test", "target_val", "target_test", "ak_side"])
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)
    for s in args.seeds:
        run_seed(args.source_exp, s, args.splits, args.final, args.force, args.workers, args.runs_dir)


if __name__ == "__main__":
    main()
