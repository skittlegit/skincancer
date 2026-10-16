"""Train one run. Modes: source (E1, E3, E4), finetune (E5), coral (E7).

    python -m skincancer.train --config configs/e1_effb0.yaml --seed 0
    python -m skincancer.train --config configs/e5_finetune.yaml --seed 0 --set data.fraction=25

Writes runs/<name>/s<seed>/: best.pt, config.yaml, history.csv, train_prior.json,
summary.json, preds/<split>.csv for every split in eval.splits.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import runs
from .config import load_config
from .labels import indian_prior
from .metrics import macro_auc
from .models import coral_loss, param_groups, set_trainable, train_mode


def select_data(manifest: pd.DataFrame, cfg: dict):
    """-> (train labelled, val labelled, unlabelled target or None)  following D1."""
    mode = cfg["mode"]
    if mode in ("source", "coral"):
        train = runs.labelled(runs.split_rows(manifest, "source_train"), cfg)
        if cfg["data"].get("extra_train"):   # D16: training-only extra data, outside the manifest
            from .data.extra import load_extra
            extra = load_extra(cfg["data"]["extra_train"], cfg["data"]["label_col"], runs.classes_of(cfg))
            train = pd.concat([train, extra[train.columns.intersection(extra.columns)]], ignore_index=True)
        val = runs.labelled(runs.split_rows(manifest, "source_val"), cfg)
    elif mode == "finetune":
        train = runs.split_rows(manifest, "target_train")
        frac = cfg["data"]["fraction"]
        if frac < 100:
            ids = pd.read_csv(f"{cfg['splits_dir']}/e5/target_train_f{frac}_s{cfg['seed']}.csv")["image_id"]
            train = train[train["image_id"].isin(set(ids))]
        train = runs.labelled(train, cfg)
        val = runs.labelled(runs.split_rows(manifest, "target_val"), cfg)
    else:
        raise ValueError(f"unknown mode {mode}")
    unl = runs.split_rows(manifest, "target_train") if mode == "coral" else None
    return train, val, unl


def class_weights(counts: np.ndarray, cfg: dict, classes: list[str]):
    """-> (loss weights, effective prior the trained model behaves as if it saw)  (D7)."""
    t = cfg["train"]
    pi = counts / counts.sum()
    if t["class_weighting"] == "none":
        return np.ones(len(counts)), pi
    if t["class_weighting"] == "balanced":
        target = np.full(len(counts), 1.0 / len(counts))
    elif t["class_weighting"] == "indian":
        target = indian_prior(classes, np.full(len(counts), 1.0 / len(counts)), t["prior_mix"])
    else:
        raise ValueError(t["class_weighting"])
    w = target / np.maximum(pi, 1e-12)
    return w / (w * pi).sum(), target


def evaluate_loader(model, loader) -> float:
    model.eval()
    P = runs.infer(model, loader)
    return macro_auc(loader.dataset.labels, P)


def train(cfg: dict) -> dict:
    runs.set_seed(cfg["seed"])
    rdir = runs.run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, rdir / "config.yaml")
    dev = runs.device()
    classes = runs.classes_of(cfg)
    manifest = pd.read_csv(cfg["manifest"])
    train_df, val_df, unl_df = select_data(manifest, cfg)
    label_col = cfg["data"]["label_col"]
    counts = train_df[label_col].value_counts().reindex(classes, fill_value=0).to_numpy(float)
    if (counts == 0).any():
        raise ValueError(f"empty class in training data: {dict(zip(classes, counts))}")
    weights, eff_prior = class_weights(counts, cfg, classes)
    runs.write_json({"classes": classes, "counts": counts.tolist(), "loss_weights": weights.tolist(),
                     "effective_prior": eff_prior.tolist()}, rdir / "train_prior.json")
    print(f"[{cfg['name']} s{cfg['seed']}] mode={cfg['mode']} train={len(train_df)} val={len(val_df)}"
          + (f" unlabelled={len(unl_df)}" if unl_df is not None else "") + f" device={dev}")

    model = runs.build_model(cfg)
    if cfg.get("init_from"):
        model.load_state_dict(torch.load(cfg["init_from"], map_location="cpu"))
    model.to(dev)
    # Drop the last partial batch, unless the whole set is smaller than one batch (E5 subsets).
    tr_loader = runs.make_loader(train_df, cfg, train=True, seed=cfg["seed"],
                                 drop_last=len(train_df) >= cfg["train"]["batch_size"])
    steps = cfg["train"].get("steps_per_epoch")   # D18: fixed optimizer steps per epoch (None = one pass)
    if steps == "auto":   # one pass over the *full* target_train, whatever fraction is used (D21)
        n_full = len(runs.labelled(runs.split_rows(manifest, "target_train"), cfg))
        steps = math.ceil(n_full / cfg["train"]["batch_size"])
    tr_iter = runs.forever(tr_loader) if steps else None
    val_loader = runs.make_loader(val_df, cfg, train=False)
    tval_loader = None
    unl_iter = None
    if cfg["mode"] == "coral":
        unl_iter = runs.forever(runs.make_loader(unl_df, cfg, train=True, seed=cfg["seed"] + 1, drop_last=True))
        tval = runs.labelled(runs.split_rows(manifest, "target_val"), cfg)
        tval_loader = runs.make_loader(tval, cfg, train=False) if len(tval) else None
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=dev))
    use_amp = cfg["train"]["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lam = float(cfg["train"].get("coral_lambda", 0.0))

    wb = None
    if cfg.get("wandb", {}).get("enabled"):
        import wandb
        wb = wandb.init(project=cfg["wandb"]["project"], name=f"{cfg['name']}/s{cfg['seed']}", config=cfg)

    history, best, epoch_global = [], {"val_auc": -math.inf}, 0
    for si, stage in enumerate(cfg["train"]["stages"]):
        set_trainable(model, stage["unfreeze"])
        opt = torch.optim.AdamW(param_groups(model, stage["lr_head"], stage["lr_backbone"],
                                             cfg["train"]["weight_decay"]))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(stage["epochs"], 1))
        stale = 0
        for ep in range(stage["epochs"]):
            t0 = time.time()
            train_mode(model)
            tot, tot_coral, n = 0.0, 0.0, 0
            for x, y in (itertools.islice(tr_iter, steps) if steps else tr_loader):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with torch.autocast(dev.type, enabled=use_amp):
                    logits, fs = model(x, return_features=True)
                    loss = criterion(logits, y)
                    if unl_iter is not None and lam > 0:
                        xt, _ = next(unl_iter)
                        _, ft = model(xt.to(dev, non_blocking=True), return_features=True)
                        lc = coral_loss(fs, ft)
                        loss = loss + lam * lc
                        tot_coral += lc.item() * len(y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(y)
                n += len(y)
            sched.step()
            val_auc = evaluate_loader(model, val_loader)
            rec = {"epoch": epoch_global, "stage": si, "loss": tot / max(n, 1), "coral": tot_coral / max(n, 1),
                   "val_auc": val_auc, "secs": round(time.time() - t0, 1)}
            if tval_loader is not None:
                rec["target_val_auc"] = evaluate_loader(model, tval_loader)  # logged, never used to stop (D1)
            history.append(rec)
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            if wb:
                wb.log(rec)
            score = -math.inf if np.isnan(val_auc) else val_auc
            if score > best["val_auc"]:
                best = dict(rec, val_auc=score)
                torch.save(model.state_dict(), rdir / "best.pt")
                stale = 0
            else:
                stale += 1
            epoch_global += 1
            if stage.get("early_stop") and stale >= stage.get("patience", 5):
                break
    if not (rdir / "best.pt").exists():
        torch.save(model.state_dict(), rdir / "best.pt")
    pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
    runs.write_json({"best": best, "n_train": len(train_df), "n_val": len(val_df)}, rdir / "summary.json")

    # Shut down the training loaders' persistent workers before predicting: with CORAL's
    # four loaders the extra worker pools ran a 15 GB machine out of memory (segfault).
    del tr_loader, val_loader, tval_loader, unl_iter
    if "tr_iter" in locals():
        del tr_iter
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
    if wb:
        wb.finish()
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
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
