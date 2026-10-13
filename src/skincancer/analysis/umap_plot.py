"""UMAP of frozen embeddings, coloured by domain and by class: do points cluster by
disease or by dataset?

    python -m skincancer.analysis.umap_plot --run runs/e1_effb0/s0 --splits source_test target_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .. import predict  # noqa: E402
from ..report import INK_2, MUTED, SERIES  # noqa: E402

# Three cancers take the three all-pairs-safe slots; benign is gray ("the rest").
CLASS_COLORS = {"bcc": SERIES[0], "scc": SERIES[1], "nmsc": SERIES[0], "mel": SERIES[2], "benign": MUTED}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", default=["source_test", "target_test"])
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    import umap  # heavy; imported only here

    splits = args.splits[:3]  # at most 3 domains (palette cap for scatter)
    rdir = Path(args.run)
    cfg = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    meta = pd.read_csv(cfg["manifest"]).set_index("image_id")
    ids, feats, dom = [], [], []
    for s in splits:
        i, f = predict.embeddings(rdir, s)
        ids.append(i)
        feats.append(f)
        dom += [s] * len(i)
    ids, F, dom = np.concatenate(ids), np.vstack(feats), np.array(dom)
    Z = umap.UMAP(random_state=0).fit_transform(F)
    labels = meta.loc[ids, cfg["data"]["label_col"]].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), dpi=200)
    panels = [("domain", dom, dict(zip(splits, SERIES))),
              ("class", labels, {c: CLASS_COLORS[c] for c in pd.unique(labels) if c in CLASS_COLORS})]
    for ax, (title, values, palette) in zip(axes, panels):
        for v, c in palette.items():
            m = values == v
            ax.scatter(Z[m, 0], Z[m, 1], s=8, color=c, label=f"{v} ({m.sum()})", alpha=0.75,
                       edgecolors="#fcfcfb", linewidths=0.3)
        ax.set_title(f"coloured by {title}", fontsize=9, color=INK_2)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.legend(frameon=False, fontsize=7, markerscale=2)
    fig.tight_layout()
    out = Path(args.out or f"reports/figures/umap_{cfg['name']}_s{cfg['seed']}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
