"""E3 image processing: Shades-of-Gray colour constancy, CLAHE on CIELAB L only, and a
lesion-centred crop. All functions take and return uint8 RGB arrays."""
from __future__ import annotations

import cv2
import numpy as np


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Finlayson & Trezzi (2004). Rescales channels so the Minkowski p-mean is grey;
    no per-channel stretching, so pigment contrast is kept."""
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, p), axis=(0, 1)), 1.0 / p)
    illum = illum / (np.linalg.norm(illum) + 1e-6)
    out = x / (illum * np.sqrt(3.0) + 1e-6)
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_l(img: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel of CIELAB only; a and b (colour) are left untouched."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def lesion_crop(img: np.ndarray, margin: float = 0.15, ring: float = 0.1,
                min_area: float = 0.002, max_area: float = 0.9) -> tuple[np.ndarray, bool]:
    """Crop to the region most different in LAB from the surrounding skin.

    Skin colour = median of an outer ring of the image. Pixels that are not skin-like
    (rulers / paper: bright and colourless; background, clothing, hair: very dark) are
    removed first, and a larger opening drops thin hairs. The distance map is
    Otsu-thresholded; the component scoring highest on area x closeness to centre wins.
    Falls back to a central 80% crop (returns ok=False) when nothing plausible is found.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (256, int(256 * h / w))) if w > 256 else img
    sh, sw = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, chroma = lab[..., 0], np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    # OpenCV LAB: L in 0..255, a/b centred at 128. Rulers/paper are masked anywhere; dark
    # regions only if they touch the border (background, clothing), so a very dark lesion
    # in the middle of the image is never masked.
    paper = (L > 200) & (chroma < 12)
    n_d, lab_d = cv2.connectedComponents((L < 25).astype(np.uint8))
    edge_ids = set(np.unique(np.r_[lab_d[0], lab_d[-1], lab_d[:, 0], lab_d[:, -1]])) - {0}
    dark_bg = np.isin(lab_d, list(edge_ids)) if edge_ids else np.zeros_like(paper)
    not_skin = cv2.dilate((paper | dark_bg).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    r = max(1, int(ring * min(sh, sw)))
    mask_ring = np.zeros((sh, sw), bool)
    mask_ring[:r], mask_ring[-r:], mask_ring[:, :r], mask_ring[:, -r:] = True, True, True, True
    ring_skin = mask_ring & ~not_skin
    skin = np.median(lab[ring_skin if ring_skin.sum() > 50 else mask_ring], axis=0)
    dist = np.sqrt(((lab - skin) ** 2 * np.array([0.5, 1.0, 1.0])).sum(-1))
    dist[not_skin] = 0.0
    dist = cv2.GaussianBlur(dist, (0, 0), 3)
    d8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary)

    best, best_score = None, 0.0
    diag = np.hypot(sh, sw)
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA] / (sh * sw)
        if not (min_area <= area <= max_area):
            continue
        closeness = 1.0 - np.hypot(cents[k][0] - sw / 2, cents[k][1] - sh / 2) / (diag / 2)
        score = area * max(closeness, 0.0)
        if score > best_score:
            best, best_score = k, score
    if best is None:
        return center_crop(img, 0.8), False

    x, y, bw, bh = stats[best, :4]
    scale = w / sw
    cx, cy = (x + bw / 2) * scale, (y + bh / 2) * scale
    side = max(bw, bh) * scale * (1 + 2 * margin)
    side = min(side, min(h, w))
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    s = int(side)
    return img[y0:y0 + s, x0:x0 + s], True


def center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    s = int(min(h, w) * frac)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def apply(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Run the enabled steps in a fixed order: colour constancy -> CLAHE -> crop."""
    info = {"crop_ok": None}
    if cfg.get("color_constancy"):
        img = shades_of_gray(img)
    if cfg.get("clahe"):
        img = clahe_l(img)
    if cfg.get("lesion_crop"):
        img, info["crop_ok"] = lesion_crop(img)
    return img, info


def tag(cfg: dict) -> str:
    """Short cache key for a preprocessing setting, e.g. 'cc-clahe-crop' or 'raw'."""
    parts = [n for k, n in (("color_constancy", "cc"), ("clahe", "clahe"), ("lesion_crop", "crop")) if cfg.get(k)]
    return "-".join(parts) or "raw"


def main(argv=None):
    """Crop check (work queue item 4): failure rate over all used images + a contact sheet
    of N random crops to inspect by eye.

        python -m skincancer.preprocess --n 100
    """
    import argparse
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from .config import load_data_config
    from .data.dataset import read_rgb

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="reports/figures/crop_check.png")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    m = pd.read_csv(cfg["manifest"])
    m = m[m["split"].fillna("excluded") != "excluded"]
    # Contact sheet from training splits only: tuning the crop by looking at test images
    # would be test-set peeking, and DDI's terms forbid showing its images.
    train_m = m[m["split"].isin(["source_train", "target_train"])]
    root = Path(cfg["data_root"])
    rows = []
    for r in m.itertuples():
        _, ok = lesion_crop(read_rgb(root / r.path))
        rows.append({"image_id": r.image_id, "dataset": r.dataset, "fst_bucket": r.fst_bucket, "crop_ok": ok})
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/crop_check.csv", index=False)
    print("crop failure rate (fallback to centre crop):")
    print((1 - res.groupby(["dataset", "fst_bucket"])["crop_ok"].mean()).round(3).to_string())
    sample = train_m.sample(min(args.n, len(train_m)), random_state=0)
    cols = 10
    rows_n = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 1.4, rows_n * 1.5), dpi=120, squeeze=False)
    for ax, r in zip(axes.flat, sample.itertuples()):
        crop, ok = lesion_crop(read_rgb(root / r.path))
        ax.imshow(crop)
        ax.set_title(("" if ok else "FALLBACK ") + str(r.fst_bucket), fontsize=6)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
