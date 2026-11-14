"""Self-supervised pretraining of SkinNet (SimCLR) on unlabelled photos, no outside weights.

Written for idea L (DECISIONS.md, section 17), discarded on 14 Nov 2026; kept so pretraining can
be tried again. A config needs a `pretrain:` block and `model.init` pointing at the result (idea
L's config is in git history, tag checkpoint-4: configs/skinnet-idea-l.yaml).

    python -m skincancer.pretrain --config configs/<name>.yaml

Two random views of each photo go through SkinNet and a small projection head; the loss
(NT-Xent) pulls the two views of one photo together and pushes every other photo in the batch
away. No label is read. The photos are the ones a model may train on (`pretrain.splits` of the
manifest and the `pretrain.extra` sets of data/extra/, which already exclude near-duplicates of
any validation or test image); no validation or test image is ever used.

Writes runs/<name>/pretrain/: backbone.pt (SkinNet weights, no head), history.csv,
summary.json, config.yaml. `train` starts from backbone.pt when `model.init` names it.
"""
from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from . import runs
from .config import load_config
from .data.dataset import MEAN, STD
from .models import SkinNet
from .train import make_scheduler

EVAL_SPLITS_NEVER = {"val", "test", "midas_val", "midas_test", "ddi_test", "dermacon_test", "dermacon_other", "ak_side"}


def ssl_transform(img_size: int) -> A.Compose:
    """SimCLR views. Colour is changed in brightness, contrast and saturation only: hue would
    change apparent skin tone (as in dataset.train_transform)."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.2, 1.0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.0, p=0.8),
        A.ToGray(p=0.1),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


class TwoViews(Dataset):
    def __init__(self, df: pd.DataFrame, loader, transform):
        self.df = df.reset_index(drop=True)
        self.loader = loader
        self.tf = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = self.loader(row["image_id"], row["path"])
        return self.tf(image=img)["image"], self.tf(image=img)["image"]


class SimCLR(nn.Module):
    def __init__(self, width: int, depths, proj_dim: int = 128):
        super().__init__()
        self.backbone = SkinNet(width, depths)
        d = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(d, 2 * d), nn.BatchNorm1d(2 * d), nn.ReLU(inplace=True),
                                  nn.Linear(2 * d, proj_dim))

    def forward(self, x):
        return self.proj(self.backbone(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """SimCLR loss over 2N views: each view's positive is the other view of its photo."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def pretrain_rows(cfg: dict) -> pd.DataFrame:
    """Every photo a model may train on, labels ignored."""
    p = cfg["pretrain"]
    bad = EVAL_SPLITS_NEVER.intersection(p["splits"])
    if bad:
        raise ValueError(f"pretrain.splits may not include evaluation splits: {sorted(bad)}")
    manifest = pd.read_csv(cfg["manifest"], low_memory=False)
    parts = [runs.split_rows(manifest, p["splits"])[["image_id", "path", "dataset"]]]
    for name in p.get("extra", []):
        from .data.extra import EXTRA_DIR, EXTRA_SPLIT
        ex = pd.read_csv(Path(EXTRA_DIR) / f"{name}.csv")
        parts.append(ex.loc[ex["split"] == EXTRA_SPLIT, ["image_id", "path", "dataset"]])
    df = pd.concat(parts, ignore_index=True).drop_duplicates("image_id").reset_index(drop=True)
    return df


def pretrain(cfg: dict) -> Path:
    p = cfg["pretrain"]
    runs.set_seed(p.get("seed", 0))
    out = Path(cfg["runs_dir"]) / cfg["name"] / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    runs.save_config(cfg, out / "config.yaml")
    dev = runs.device()
    df = pretrain_rows(cfg)
    print(f"[{cfg['name']} pretrain] {len(df)} photos: {df['dataset'].value_counts().to_dict()} device={dev}", flush=True)

    # two views of 256 photos is ~300 MB a batch; each worker queues two, so fewer workers than
    # training keeps Windows' shared memory (paging file) from running out (error 1455)
    nw = p.get("num_workers", cfg["train"]["num_workers"])
    if nw > 0:
        import os
        n = torch.get_num_threads()
        os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        torch.set_num_threads(n)
    ds = TwoViews(df, runs.image_loader(cfg), ssl_transform(p.get("img_size", cfg["data"]["img_size"])))
    g = torch.Generator().manual_seed(p.get("seed", 0))
    loader = DataLoader(ds, batch_size=p["batch_size"], shuffle=True, drop_last=True, num_workers=nw, generator=g,
                        pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
                        worker_init_fn=partial(runs._seed_worker, base=p.get("seed", 0)))   # picklable: Windows spawns workers
    m = cfg["model"]
    model = SimCLR(m["width"], m["depths"], p.get("proj_dim", 128)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = make_scheduler(opt, p["epochs"], p["warmup_epochs"])
    use_amp = cfg["train"]["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    for ep in range(p["epochs"]):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x1, x2 in loader:
            x1, x2 = x1.to(dev, non_blocking=True), x2.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=use_amp):
                z = model(torch.cat([x1, x2]))
            loss = nt_xent(z[: len(x1)], z[len(x1):], p["temperature"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x1)
            n += len(x1)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        rec = {"epoch": ep, "lr": lr, "loss": tot / max(n, 1), "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if (ep + 1) % 20 == 0 or ep + 1 == p["epochs"]:   # the last epoch's weights are the ones used
            torch.save(model.backbone.state_dict(), out / "backbone.pt")
    runs.write_json({"n_photos": len(df), "by_dataset": df["dataset"].value_counts().to_dict(),
                     "epochs": p["epochs"], "final_loss": history[-1]["loss"] if history else math.nan}, out / "summary.json")
    return out / "backbone.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, list(args.set))
    done = Path(cfg["runs_dir"]) / cfg["name"] / "pretrain" / "summary.json"
    if done.exists() and not args.overwrite:
        print(f"{done.parent} already finished; pass --overwrite to redo")
        return
    pretrain(cfg)


if __name__ == "__main__":
    main()
