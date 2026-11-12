"""Metrics with CIs for one experiment, from its prediction files only.

    python -m skincancer.evaluate --exp skinnet

Reads runs/<exp>/s*/preds/<split>.csv; writes results/<exp>/metrics.csv in long
format: exp, eval_set, subgroup_col, subgroup, metric, mean, std, ci_lo, ci_hi, n, n_malignant, n_seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics as M
from .labels import CLASSES

# eval set -> (task, subgroup columns). Subgroups by type of lesion show which cancers are
# missed and which harmless conditions raise false alarms.
EVAL_SETS = {
    "val": ("binary", ["dataset", "label_fine"]),
    "test": ("binary", ["dataset", "label_fine"]),
    "ddi_test": ("binary", []),                   # biopsy-proven, Stanford
    "midas_test": ("binary", ["label_fine"]),     # biopsy-proven clinical photos, Stanford
    "dermacon_test": ("binary", ["label_fine"]),  # Indian tumours
    "ak_side": ("rates", []),                     # precancerous: neither class
    "dermacon_other": ("rates", []),              # Indian clinic images that are not cancer
}
# Cancers and the benign tumours that resemble them: the hard part of the task.
HARD_TYPES = {"bcc", "scc", "mel", "rare_cancer", "benign"}
MIN_SUBGROUP = 10
OP_SENSITIVITY = 0.90   # operating point: threshold fixed on OP_SPLIT, applied to every set
OP_SPLIT = "val"


def load_preds(exp_dir: Path, split: str) -> tuple[list[int], list[pd.DataFrame]]:
    seeds, frames = [], []
    for sdir in sorted(exp_dir.glob("s*")):
        f = sdir / "preds" / f"{split}.csv"
        if f.exists():
            seeds.append(int(sdir.name[1:]))
            frames.append(pd.read_csv(f))
    return seeds, frames


def run_config(exp_dir: Path) -> dict:
    """Resolved config of the first seed."""
    sdirs = sorted(exp_dir.glob("s*"))
    if not sdirs:
        raise FileNotFoundError(f"no runs under {exp_dir}")
    return yaml.safe_load((sdirs[0] / "config.yaml").read_text(encoding="utf-8"))


def score_set(split: str, frames: list[pd.DataFrame], manifest: pd.DataFrame, classes: list[str],
              label_col: str, n_boot: int, thresholds: list[float] | None = None) -> list[dict]:
    task, sub_cols = EVAL_SETS.get(split, ("multiclass", []))
    ids = frames[0]["image_id"]
    for f in frames[1:]:
        assert set(f["image_id"]) == set(ids), f"{split}: seeds scored different images"
    meta = manifest.set_index("image_id").loc[ids].reset_index()
    Ps = [f.set_index("image_id").loc[ids, [f"p_{c}" for c in classes]].to_numpy() for f in frames]
    idx = {c: i for i, c in enumerate(classes)}
    y = meta[label_col].map(idx).fillna(-1).astype(int).to_numpy()
    y_bin = meta["label_binary"].astype(float).to_numpy()
    groups = meta["group_id"].astype(str).to_numpy()
    rows = []

    if task == "rates":
        # Share of images called malignant, and called each class (these sets hold no cancers).
        fns = {"pred_malignant_rate": lambda y_, yb, P: float(M.pred_malignant(P, classes).mean())}
    else:
        fns = M.metric_fns(classes)
        if meta["dataset"].nunique() > 1:
            # Mean of the AUCs inside each dataset: immune to the photo-type shortcut. The
            # bootstrap hands every function a resampled `y`; here that slot carries the dataset.
            y = pd.factorize(meta["dataset"])[0]
            fns["source_auc"] = lambda src, yb, P: M.source_auc(
                yb[~np.isnan(yb)], M.p_malignant(P, classes)[~np.isnan(yb)], src[~np.isnan(yb)])

    subsets = [("all", "all", np.ones(len(meta), bool))]
    for col in sub_cols:
        for val in sorted(meta[col].dropna().unique(), key=str):
            subsets.append((col, str(val), (meta[col] == val).to_numpy()))
    if task != "rates" and split in ("val", "test"):
        subsets.append(("hard", "cancer_vs_lookalike", meta["label_fine"].isin(HARD_TYPES).to_numpy()))
    op = None
    if thresholds is not None and not np.any(np.isnan(thresholds)):
        op = [(M.p_malignant(P, classes) - t)[:, None] for P, t in zip(Ps, thresholds)]
        op_fns = M.operating_point_fns()
        if task == "rates":
            op_fns = {"pred_malignant_rate_op90": lambda y_, yb, P: float((P[:, 0] >= 0).mean())}
    for col, val, mask in subsets:
        if mask.sum() < MIN_SUBGROUP:
            continue
        res = M.bootstrap(fns, y[mask], y_bin[mask], [P[mask] for P in Ps], groups[mask], n_boot=n_boot)
        if op is not None:   # same seed, so the same resample indices as above
            res.update(M.bootstrap(op_fns, y[mask], y_bin[mask], [Q[mask] for Q in op], groups[mask], n_boot=n_boot))
        for metric, r in res.items():
            rows.append({"eval_set": split, "subgroup_col": col, "subgroup": val, "metric": metric,
                         "mean": r["mean"], "std": r["std"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "n": int(mask.sum()), "n_malignant": int(np.nansum(y_bin[mask] == 1)),
                         "n_seeds": len(Ps)})
    return rows


def evaluate(exp: str, runs_dir="runs", results_dir="results", manifest_path=None,
             n_boot: int = 1000, splits=None) -> pd.DataFrame:
    exp_dir = Path(runs_dir) / exp
    cfg = run_config(exp_dir)
    label_col = cfg["data"]["label_col"]
    classes = CLASSES[label_col]
    cols = [f"p_{c}" for c in classes]
    manifest = pd.read_csv(manifest_path or cfg.get("manifest", "data/manifest.csv"))

    # Per-seed threshold on P(malignant) that catches OP_SENSITIVITY of the cancers in OP_SPLIT.
    thresholds = {}
    seeds, frames = load_preds(exp_dir, OP_SPLIT)
    yb = manifest.set_index("image_id")["label_binary"]
    for seed, f in zip(seeds, frames):
        thresholds[seed] = M.sensitivity_threshold(yb.loc[f["image_id"]].to_numpy(),
                                                   M.p_malignant(f[cols].to_numpy(), classes), OP_SENSITIVITY)
    rows = []
    for split in splits or EVAL_SETS:
        seeds, frames = load_preds(exp_dir, split)
        if not frames or frames[0].empty:
            continue
        rows += score_set(split, frames, manifest, classes, label_col, n_boot,
                          [thresholds.get(s, np.nan) for s in seeds])
    out = pd.DataFrame(rows)
    out.insert(0, "exp", exp)
    odir = Path(results_dir) / exp
    odir.mkdir(parents=True, exist_ok=True)
    # a partial evaluation (chosen splits only) never overwrites the full table
    out.to_csv(odir / ("metrics.csv" if splits is None else f"metrics_{'_'.join(splits)}.csv"), index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, nargs="+")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--splits", nargs="+", help="score only these (e.g. val, when choosing between models)")
    args = ap.parse_args(argv)
    for exp in args.exp:
        out = evaluate(exp, args.runs_dir, args.results_dir, n_boot=args.n_boot, splits=args.splits)
        head = out[(out["subgroup"] == "all")
                   & out["metric"].isin(["binary_auc", "source_auc", "missed_cancer_rate_op90", "false_alarm_rate_op90"])]
        print(head[["exp", "eval_set", "metric", "mean", "ci_lo", "ci_hi", "n", "n_malignant", "n_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
