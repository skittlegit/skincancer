"""Grad-CAM grid: N correct and N wrong predictions from one split, overlaid on the image.
Look for attention on rulers, markings, borders or background.

    python -m skincancer.analysis.gradcam --run runs/e1_effb0/s0 --split target_test --n 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .. import runs  # noqa: E402
from ..data.dataset import MEAN, STD, SkinDataset, eval_transform  # noqa: E402
from ..models import target_layer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="target_test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    rdir = Path(args.run)
    model, cfg = runs.load_run(rdir)
    classes = runs.classes_of(cfg)
    preds = pd.read_csv(rdir / "preds" / f"{args.split}.csv")
    meta = pd.read_csv(cfg["manifest"]).set_index("image_id")
    preds["true"] = meta.loc[preds["image_id"], cfg["data"]["label_col"]].to_numpy()
    preds["pred"] = np.array(classes)[preds[[f"p_{c}" for c in classes]].to_numpy().argmax(1)]
    preds = preds[preds["true"].isin(classes)]
    rng = np.random.default_rng(args.seed)
    picks = []
    for correct in (True, False):
        pool = preds[(preds["true"] == preds["pred"]) == correct]
        picks.append(pool.iloc[rng.permutation(len(pool))[:args.n]].assign(correct=correct))
    sel = pd.concat(picks).reset_index(drop=True)
    df = meta.loc[sel["image_id"]].reset_index()
    d = cfg["data"]
    ds = SkinDataset(df, runs.image_loader(cfg), eval_transform(d["img_size"], d["resize_short"]))
    dev = next(model.parameters()).device

    cam = GradCAM(model=model, target_layers=[target_layer(model)])
    cols = 10
    nrows = max(1, int(np.ceil(len(sel) / cols)))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 1.6, nrows * 1.85), dpi=150, squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        if k >= len(sel):
            continue
        x, _ = ds[k]
        target = ClassifierOutputTarget(classes.index(sel.loc[k, "pred"]))
        heat = cam(input_tensor=x[None].to(dev), targets=[target])[0]
        img = (x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1).astype(np.float32)
        ax.imshow(show_cam_on_image(img, heat, use_rgb=True))
        mark = "ok" if sel.loc[k, "correct"] else "WRONG"
        ax.set_title(f"{mark} {sel.loc[k, 'true']}→{sel.loc[k, 'pred']}\n{df.loc[k, 'fst_bucket']}", fontsize=6)
    fig.suptitle(f"Grad-CAM, {cfg['name']} s{cfg['seed']} on {args.split}: correct first, then wrong", fontsize=9)
    fig.tight_layout()
    out = Path(args.out or f"reports/figures/gradcam_{cfg['name']}_s{cfg['seed']}_{args.split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
