"""Torch dataset over manifest rows, with an on-disk cache of preprocessed images, and
the augmentation pipelines (no hue jitter: hue changes apparent skin tone)."""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from .. import preprocess

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def eval_transform(img_size: int, resize_short: int) -> A.Compose:
    return A.Compose([
        A.SmallestMaxSize(max_size=resize_short),
        A.CenterCrop(img_size, img_size),
        A.Normalize(MEAN, STD),
        ToTensorV2(),
    ])


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class ImageLoader:
    """Loads an image with preprocessing applied, cached as JPEG under cache_dir/<tag>/."""

    def __init__(self, data_root, preprocess_cfg: dict, cache_dir=None, max_side: int = 512):
        self.root = Path(data_root)
        self.cfg = preprocess_cfg or {}
        self.tag = preprocess.tag(self.cfg)
        self.cache = Path(cache_dir) / self.tag if cache_dir else None
        self.max_side = max_side

    def __call__(self, image_id: str, rel_path: str) -> np.ndarray:
        if self.cache is not None:
            cached = self.cache / f"{image_id}.jpg"
            if cached.exists():
                return read_rgb(cached)
        img = read_rgb(self.root / rel_path)
        img, _ = preprocess.apply(img, self.cfg)
        h, w = img.shape[:2]
        if max(h, w) > self.max_side:
            s = self.max_side / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.cache / f"{image_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img


class SkinDataset(Dataset):
    """Returns (image tensor, class index). Index is -1 for rows whose label is not one of
    `classes` (unlabelled target batches for CORAL, or binary-only test sets)."""

    def __init__(self, df: pd.DataFrame, loader: ImageLoader, transform, classes=None, label_col=None):
        self.df = df.reset_index(drop=True)
        self.loader = loader
        self.transform = transform
        idx = {c: i for i, c in enumerate(classes or [])}
        labels = self.df[label_col] if label_col in self.df else pd.Series([None] * len(self.df))
        self.labels = np.array([idx.get(l, -1) for l in labels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = self.loader(row["image_id"], row["path"])
        return self.transform(image=img)["image"], torch.tensor(self.labels[i])
