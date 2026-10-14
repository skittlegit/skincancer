"""Summary tables and the report figures, from results/*/metrics.csv and prediction files.

    python -m skincancer.report tables  --ref e1_effb0
    python -m skincancer.report gap     --exps e1_effb0 e3_all e4_indian_loss e6_fusion e7_coral_l1.0 --out gap.png
    python -m skincancer.report curve   --ref e1_effb0 --prefix e5_f
    python -m skincancer.report confusion --exp e1_effb0 --split target_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import metrics as M  # noqa: E402
from .evaluate import load_preds, run_config  # noqa: E402
from .labels import CLASSES  # noqa: E402

# Reference palette (dataviz skill): first three categorical slots validate all-pairs.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# (eval_set, subgroup_col, subgroup, metric, label) — the headline target numbers.
TARGETS = [
    ("target_test", "all", "all", "macro_auc", "Fitzpatrick17k IV–VI (macro AUC)"),
    ("ddi_test", "fst_bucket", "V-VI", "binary_auc", "DDI V–VI (binary AUC)"),
    ("dermacon_test", "all", "all", "binary_auc", "DermaCon-IN (binary AUC)"),
    ("midas_test", "fst_bucket", "I-III", "macro_auc", "MIDAS I–III (macro AUC)"),
]
FIG_DIR = Path("reports/figures")


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_results(results_dir="results") -> pd.DataFrame:
    files = list(Path(results_dir).glob("*/metrics.csv"))
    if not files:
        raise SystemExit(f"no metrics under {results_dir}; run skincancer.evaluate first")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def pick(res, exp, eval_set, col, sub, metric):
    r = res[(res["exp"] == exp) & (res["eval_set"] == eval_set) & (res["subgroup_col"] == col)
            & (res["subgroup"].astype(str) == sub) & (res["metric"] == metric)]
    return r.iloc[0] if len(r) else None


def _has(r) -> bool:
    return r is not None and not np.isnan(r["mean"])


def tables(res: pd.DataFrame, ref: str, out: Path) -> pd.DataFrame:
    """One row per exp x headline set: mean [CI], plus gap = ref in-domain AUC - target AUC."""
    rows = []
    for exp in sorted(res["exp"].unique()):
        for es, col, sub, metric, label in [("source_test", "all", "all", "macro_auc", "Source test (macro AUC)"),
                                            *TARGETS]:
            r = pick(res, exp, es, col, sub, metric)
            if r is None:
                continue
            ref_r = pick(res, ref, "source_test", "all", "all", metric)
            gap = ref_r["mean"] - r["mean"] if ref_r is not None and es != "source_test" else np.nan
            rows.append({"exp": exp, "set": label, "mean": r["mean"], "std": r["std"],
                         "ci": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", "gap_vs_ref": gap,
                         "n": r["n"], "n_malignant": r["n_malignant"], "seeds": r["n_seeds"]})
        for es in ("target_test", "ddi_test"):
            for metric in ("missed_cancer_rate", "false_alarm_rate"):
                r = pick(res, exp, es, "all", "all", metric)
                if r is not None:
                    rows.append({"exp": exp, "set": f"{es} {metric}", "mean": r["mean"], "std": r["std"],
                                 "ci": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", "gap_vs_ref": np.nan,
                                 "n": r["n"], "n_malignant": r["n_malignant"], "seeds": r["n_seeds"]})
        r = pick(res, exp, "dermacon_other", "all", "all", "pred_malignant_rate")
        if r is not None:
            rows.append({"exp": exp, "set": "DermaCon-IN non-cancer called malignant", "mean": r["mean"],
                         "std": r["std"], "ci": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", "gap_vs_ref": np.nan,
                         "n": r["n"], "n_malignant": 0, "seeds": r["n_seeds"]})
    t = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    return t


def gap_chart(res: pd.DataFrame, exps: list[str], ref: str, out: Path) -> None:
    """Grouped bars: target AUC per method, with 95% CI whiskers; dashed line = in-domain AUC."""
    sets = [t for t in TARGETS if any(_has(pick(res, e, *t[:4])) for e in exps)]
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(exps) + 1.5), 3.4), dpi=200)
    style(ax)
    width = 0.8 / max(len(sets), 1)
    x = np.arange(len(exps))
    lows = [0.5]
    for i, (es, col, sub, metric, label) in enumerate(sets):
        vals = [pick(res, e, es, col, sub, metric) for e in exps]
        means = np.array([v["mean"] if v is not None else np.nan for v in vals])
        lo = np.array([v["mean"] - v["ci_lo"] if v is not None else 0 for v in vals])
        hi = np.array([v["ci_hi"] - v["mean"] if v is not None else 0 for v in vals])
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means - 0.5, width=width - 0.02, bottom=0.5, color=SERIES[i], label=label,
               edgecolor="#fcfcfb", linewidth=1)
        ax.errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        lows += list(means - lo)
    ref_r = pick(res, ref, "source_test", "all", "all", "macro_auc")
    if ref_r is not None:
        ax.axhline(ref_r["mean"], color=MUTED, linestyle="--", linewidth=1,
                   label=f"in-domain: {ref} on source test ({ref_r['mean']:.2f})")
    ax.axhline(0.5, color=AXIS, linewidth=1)  # chance; bars grow up or down from it
    ax.set_xticks(x, exps, rotation=20, ha="right")
    ax.set_ylim(max(0.0, np.floor((np.nanmin(lows) - 0.03) * 20) / 20), 1.0)
    ax.set_ylabel("AUC (95% CI)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def curve_chart(res: pd.DataFrame, ref: str, prefix: str, out: Path) -> None:
    """E5 data-size curve: target AUC vs % of target_train labels (0% = the reference run)."""
    fracs = {0: ref}
    for e in res["exp"].unique():
        if e.startswith(prefix) and e[len(prefix):].isdigit():
            fracs[int(e[len(prefix):])] = e
    xs = sorted(fracs)
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=200)
    style(ax)
    for i, (es, col, sub, metric, label) in enumerate(TARGETS[:2]):
        pts = [(f, pick(res, fracs[f], es, col, sub, metric)) for f in xs]
        pts = [(f, r) for f, r in pts if _has(r)]
        if not pts:
            continue
        f_, m_ = [p[0] for p in pts], np.array([p[1]["mean"] for p in pts])
        lo = np.array([p[1]["ci_lo"] for p in pts])
        hi = np.array([p[1]["ci_hi"] for p in pts])
        ax.fill_between(f_, lo, hi, color=SERIES[i], alpha=0.15, linewidth=0)
        ax.plot(f_, m_, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
    ref_r = pick(res, ref, "source_test", "all", "all", "macro_auc")
    if ref_r is not None:
        ax.axhline(ref_r["mean"], color=MUTED, linestyle="--", linewidth=1,
                   label=f"in-domain: {ref} on source test ({ref_r['mean']:.2f})")
    ax.set_xticks(xs)
    ax.set_xlabel("% of FST IV–VI training labels", color=INK_2, fontsize=9)
    ax.set_ylabel("AUC (95% CI)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def confusion_chart(exp: str, split: str, out: Path, runs_dir="runs", manifest_path=None, by="fst_bucket") -> None:
    """Row-normalised confusion matrix per subgroup (seed-averaged probabilities)."""
    exp_dir = Path(runs_dir) / exp
    cfg, _ = run_config(exp_dir)
    label_col = cfg["data"]["label_col"]
    classes = CLASSES[label_col]
    _, frames = load_preds(exp_dir, split)
    ids = frames[0]["image_id"]
    P = np.mean([f.set_index("image_id").loc[ids, [f"p_{c}" for c in classes]].to_numpy() for f in frames], 0)
    meta = pd.read_csv(manifest_path or cfg["manifest"]).set_index("image_id").loc[ids].reset_index()
    y = meta[label_col].map({c: i for i, c in enumerate(classes)}).fillna(-1).astype(int).to_numpy()
    subs = [("all", np.ones(len(meta), bool))] + [(str(v), (meta[by] == v).to_numpy())
                                                   for v in sorted(meta[by].dropna().unique(), key=str)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, len(subs), figsize=(2.6 * len(subs), 2.8), dpi=200)
    for ax, (name, mask) in zip(np.atleast_1d(axes), subs):
        cm = M.confusion(y[mask], P[mask], len(classes)).astype(float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                        color="#fcfcfb" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(classes)), classes, fontsize=7)
        ax.set_yticks(range(len(classes)), classes, fontsize=7)
        ax.set_title(f"{by}={name} (n={int(mask.sum())})" if name != "all" else f"all (n={int(mask.sum())})",
                     fontsize=8, color=INK_2)
        ax.set_xlabel("predicted", fontsize=7, color=MUTED)
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("true", fontsize=7, color=MUTED)
    fig.suptitle(f"{exp} on {split} (counts; shade = row %)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tables", "gap", "curve", "confusion"])
    ap.add_argument("--ref", default="e1_effb0")
    ap.add_argument("--exps", nargs="+")
    ap.add_argument("--exp")
    ap.add_argument("--split", default="target_test")
    ap.add_argument("--prefix", default="e5_f")
    ap.add_argument("--out")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.what == "confusion":
        out = Path(args.out or FIG_DIR / f"confusion_{args.exp}_{args.split}.png")
        confusion_chart(args.exp, args.split, out)
    else:
        res = load_results(args.results_dir)
        if args.what == "tables":
            out = Path(args.out or Path(args.results_dir) / "summary.csv")
            print(tables(res, args.ref, out).to_string(index=False))
        elif args.what == "gap":
            out = Path(args.out or FIG_DIR / "gap.png")
            gap_chart(res, args.exps or sorted(res["exp"].unique()), args.ref, out)
        else:
            out = Path(args.out or FIG_DIR / "e5_curve.png")
            curve_chart(res, args.ref, args.prefix, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
