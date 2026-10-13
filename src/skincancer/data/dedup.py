"""Group near-duplicate images so that splits never separate them.

Groups = union of (same patient) + (pHash Hamming distance <= threshold) + (ImageNet
embedding cosine >= threshold). Thresholds are calibrated in DECISIONS.md D10. Runs across
all datasets, so a test-only image that duplicates a training image is caught too.
Writes data/groups.csv.

    python -m skincancer.data.dedup               # ~1 min per 1k images on CPU
    python -m skincancer.data.dedup --no-embed    # pHash only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..config import load_data_config


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def phash_ints(paths: list[Path]) -> np.ndarray:
    import imagehash
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(tqdm(paths, desc="pHash")):
        with Image.open(p) as im:
            out[i] = np.uint64(int(str(imagehash.phash(im.convert("RGB"))), 16))
    return out


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def hash_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 512):
    """Yield (i, j) pairs with i < j and Hamming distance <= max_dist."""
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        dist = _popcount(block[:, None] ^ hashes[None, :])
        ii, jj = np.nonzero(dist <= max_dist)
        ii = ii + start
        keep = ii < jj
        yield from zip(ii[keep], jj[keep])


def cosine_pairs(emb: np.ndarray, min_cos: float, chunk: int = 1024):
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    for start in range(0, len(emb), chunk):
        sim = emb[start:start + chunk] @ emb.T
        ii, jj = np.nonzero(sim >= min_cos)
        ii = ii + start
        keep = ii < jj
        yield from zip(ii[keep], jj[keep])


def embed(paths: list[Path], batch_size: int = 64) -> np.ndarray:
    import timm
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg)
    feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size), desc="embed"):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
            feats.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(feats)


def group(df: pd.DataFrame, hashes: np.ndarray, max_dist: int,
          emb: np.ndarray | None = None, min_cos: float = 0.95) -> pd.DataFrame:
    uf = UnionFind(len(df))
    # same patient
    for _, idx in df.reset_index(drop=True).groupby("patient_id").groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
    n_hash = n_emb = 0
    for i, j in hash_pairs(hashes, max_dist):
        uf.union(int(i), int(j))
        n_hash += 1
    if emb is not None:
        for i, j in cosine_pairs(emb, min_cos):
            uf.union(int(i), int(j))
            n_emb += 1
    roots = np.array([uf.find(i) for i in range(len(df))])
    ids = df["image_id"].to_numpy()
    out = pd.DataFrame({"image_id": ids, "group_id": ["g_" + ids[r] for r in roots]})
    print(f"pHash pairs: {n_hash}, embedding pairs: {n_emb}, "
          f"groups: {out['group_id'].nunique()} for {len(out)} images")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    root = Path(cfg["data_root"])
    m = pd.read_csv(cfg["manifest"])
    m = m[m["exclude_reason"] != "missing_image"].reset_index(drop=True)
    paths = [root / p for p in m["path"]]
    hashes = phash_ints(paths)
    emb = None if args.no_embed else embed(paths)
    groups = group(m, hashes, cfg["dedup"]["phash_max_distance"], emb, cfg["dedup"]["embed_min_cosine"])
    groups.to_csv(cfg["groups"], index=False)
    multi = groups.groupby("group_id").size()
    print(f"{(multi > 1).sum()} groups with >1 image; wrote {cfg['groups']}")


if __name__ == "__main__":
    main()
