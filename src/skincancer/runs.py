"""Shared plumbing for training / prediction: seeding, run directories, split selection,
loaders, checkpoint loading and the prediction-file format."""
from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .data.dataset import ImageLoader, SkinDataset, eval_transform, train_transform
from .labels import CLASSES
from .models import Classifier


def _seed_worker(worker_id: int, base: int) -> None:
    np.random.seed(base * 100 + worker_id)
    random.seed(base * 100 + worker_id)


def forever(loader):
    """Endless iterator that re-iterates the loader (unlike itertools.cycle, which keeps
    every batch in memory)."""
    while True:
        yield from loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(cfg: dict) -> Path:
    return Path(cfg["runs_dir"]) / cfg["name"] / f"s{cfg['seed']}"


def classes_of(cfg: dict) -> list[str]:
    return CLASSES[cfg["data"]["label_col"]]


def split_rows(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    return manifest[manifest["split"] == split].reset_index(drop=True)


def labelled(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return df[df[cfg["data"]["label_col"]].isin(classes_of(cfg))].reset_index(drop=True)


def image_loader(cfg: dict) -> ImageLoader:
    return ImageLoader(cfg["data_root"], cfg["data"]["preprocess"], cfg.get("cache_dir"))


def make_loader(df: pd.DataFrame, cfg: dict, train: bool, seed: int = 0, drop_last: bool = False) -> DataLoader:
    d = cfg["data"]
    tf = train_transform(d["img_size"]) if train else eval_transform(d["img_size"], d["resize_short"])
    ds = SkinDataset(df, image_loader(cfg), tf, classes_of(cfg), d["label_col"])
    g = torch.Generator().manual_seed(seed)
    nw = cfg["train"]["num_workers"]
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train, drop_last=drop_last,
                      num_workers=nw, generator=g, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=partial(_seed_worker, base=seed), persistent_workers=nw > 0)


def build_model(cfg: dict, pretrained: bool | None = None) -> Classifier:
    m = cfg["model"]
    return Classifier(m["backbone"], len(classes_of(cfg)),
                      m["pretrained"] if pretrained is None else pretrained, m.get("drop", 0.2))


def save_config(cfg: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def load_run(rdir: str | Path):
    """-> (model on device, eval mode; resolved config)."""
    rdir = Path(rdir)
    cfg = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(rdir / "best.pt", map_location="cpu"))
    return model.to(device()).eval(), cfg


@torch.no_grad()
def infer(model, loader, with_features: bool = False):
    dev = next(model.parameters()).device
    probs, feats = [], []
    for x, _ in loader:
        logits, f = model(x.to(dev), return_features=True)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        if with_features:
            feats.append(f.float().cpu().numpy())
    P = np.concatenate(probs) if probs else np.zeros((0, model.head[-1].out_features))
    return (P, np.concatenate(feats)) if with_features else P


def prediction_frame(df: pd.DataFrame, P: np.ndarray, classes: list[str], cfg: dict) -> pd.DataFrame:
    """The one prediction format every analysis reads."""
    out = pd.DataFrame({"image_id": df["image_id"].to_numpy(), "exp": cfg["name"], "seed": cfg["seed"]})
    for k, c in enumerate(classes):
        out[f"p_{c}"] = P[:, k]
    return out


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
