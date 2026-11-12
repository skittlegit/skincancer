"""Train SkinNet from random weights: every layer, one learning rate, no pretrained weights.

    python -m skincancer.train --config configs/skinnet.yaml --seed 0

Writes runs/<name>/s<seed>/: best.pt, config.yaml, history.csv, summary.json and
preds/<split>.csv for every split in eval.splits.
"""
from __future__ import annotations

import argparse
import gc
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import runs
from .config import load_config
from .metrics import auc


def class_weights(counts: np.ndarray, weighting: str) -> np.ndarray:
    """Loss weight per class. balanced: every class counts equally in total, whatever its size."""
    if weighting == "none":
        return np.ones(len(counts))
    if weighting != "balanced":
        raise ValueError(f"unknown train.class_weighting {weighting!r}")
    pi = counts / counts.sum()
    w = (1.0 / len(counts)) / np.maximum(pi, 1e-12)
    return w / (w * pi).sum()


def make_scheduler(opt, epochs: int, warmup: int = 0):
    """Stepped once per epoch: linear warm-up over `warmup` epochs, then cosine decay to zero."""
    rest = max(epochs - warmup, 1)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warmup if e < warmup else 0.5 * (1 + math.cos(math.pi * (e - warmup) / rest)))


def train(cfg: dict) -> dict:
    runs.set_seed(cfg["seed"])
    rdir = runs.run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, rdir / "config.yaml")
    dev = runs.device()
    classes = runs.classes_of(cfg)
    t = cfg["train"]
    manifest = pd.read_csv(cfg["manifest"])
    train_df = runs.labelled(runs.split_rows(manifest, cfg["data"]["train_splits"]), cfg)
    val_df = runs.labelled(runs.split_rows(manifest, cfg["data"]["val_splits"]), cfg)
    label_col = cfg["data"]["label_col"]
    counts = train_df[label_col].value_counts().reindex(classes, fill_value=0).to_numpy(float)
    if (counts == 0).any():
        raise ValueError(f"empty class in training data: {dict(zip(classes, counts))}")
    weights = class_weights(counts, t["class_weighting"])

    model = runs.build_model(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']} s{cfg['seed']}] train={len(train_df)} val={len(val_df)} "
          f"params={n_params / 1e6:.2f}M device={dev}")
    tr_loader = runs.make_loader(train_df, cfg, train=True, seed=cfg["seed"], drop_last=True)
    val_loader = runs.make_loader(val_df, cfg, train=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev),
                                    label_smoothing=float(t["label_smoothing"]))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = make_scheduler(opt, t["epochs"], t["warmup_epochs"])
    use_amp = t["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, stale = [], {"val_auc": -math.inf}, 0
    for ep in range(t["epochs"]):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in tr_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        model.eval()
        val_auc = auc(val_loader.dataset.labels == 1, runs.infer(model, val_loader)[:, 1])
        rec = {"epoch": ep, "lr": lr, "loss": tot / max(n, 1), "val_auc": val_auc,
               "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
              flush=True)
        pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
        score = -math.inf if np.isnan(val_auc) else val_auc
        if score > best["val_auc"]:
            best = dict(rec, val_auc=score)
            torch.save(model.state_dict(), rdir / "best.pt")
            stale = 0
        else:
            stale += 1
        if t["patience"] and stale >= t["patience"]:
            break
    if not (rdir / "best.pt").exists():
        torch.save(model.state_dict(), rdir / "best.pt")
    runs.write_json({"best": best, "epochs_run": len(history), "n_train": len(train_df), "n_val": len(val_df),
                     "n_params": n_params, "class_counts": dict(zip(classes, counts.tolist())),
                     "loss_weights": dict(zip(classes, weights.tolist()))}, rdir / "summary.json")

    # Shut down the training loaders' worker processes before predicting (15 GB machine).
    del tr_loader, val_loader
    gc.collect()
    model.load_state_dict(torch.load(rdir / "best.pt", map_location=dev))
    model.eval()
    (rdir / "preds").mkdir(exist_ok=True)
    for split in cfg["eval"]["splits"]:
        df = runs.split_rows(manifest, split)
        if len(df) == 0:
            continue
        loader = runs.make_loader(df, cfg, train=False, persistent=False)
        P = runs.infer(model, loader, tta=cfg["eval"].get("tta", False))
        del loader
        runs.prediction_frame(df, P, classes, cfg).to_csv(rdir / "preds" / f"{split}.csv", index=False)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/skinnet.yaml")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    overrides = list(args.set) + ([f"seed={args.seed}"] if args.seed is not None else [])
    cfg = load_config(args.config, overrides)
    if (runs.run_dir(cfg) / "summary.json").exists() and not args.overwrite:
        print(f"{runs.run_dir(cfg)} already finished; pass --overwrite to retrain")
        return
    train(cfg)


if __name__ == "__main__":
    main()
