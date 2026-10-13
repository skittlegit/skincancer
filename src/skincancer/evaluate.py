"""Metrics with CIs for one experiment, from its prediction files only.

    python -m skincancer.evaluate --exp e1_effb0
    python -m skincancer.evaluate --exp e1_effb0 --prior-mix all      # E4 prior-shift sweep

Reads runs/<exp>/s*/preds/<split>.csv; writes results/<exp>[+prior_<mix>]/metrics.csv in long
format: exp, eval_set, subgroup_col, subgroup, metric, mean, std, ci_lo, ci_hi, n, n_malignant, n_seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import metrics as M
from .labels import CLASSES, INDIAN_MIXES, indian_prior

# eval set -> (task, subgroup columns). DDI / DermaCon-IN are binary-only (plan, Metrics).
EVAL_SETS = {
    "source_val": ("multiclass", []),
    "source_test": ("multiclass", ["fst_bucket", "dataset", "is_acral"]),
    "target_val": ("multiclass", []),
    "target_test": ("multiclass", ["fst_bucket", "is_acral"]),
    "pad_dark_test": ("multiclass", []),
    "ddi_test": ("binary", ["fst_bucket"]),
    "dermacon_test": ("binary", ["fst_bucket"]),
    "ak_side": ("rates", []),
    "dermacon_other": ("rates", ["fst_bucket"]),   # non-cancer Indian clinic images (D9)
}
MIN_SUBGROUP = 10


def load_preds(exp_dir: Path, split: str) -> tuple[list[int], list[pd.DataFrame]]:
    seeds, frames = [], []
    for sdir in sorted(exp_dir.glob("s*")):
        f = sdir / "preds" / f"{split}.csv"
        if f.exists():
            seeds.append(int(sdir.name[1:]))
            frames.append(pd.read_csv(f))
    return seeds, frames


def run_config(exp_dir: Path) -> tuple[dict, dict[int, np.ndarray]]:
    """-> (resolved config of the first seed, {seed: effective training prior})."""
    sdirs = sorted(exp_dir.glob("s*"))
    if not sdirs:
        raise FileNotFoundError(f"no runs under {exp_dir}")
    cfg_path = sdirs[0] / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    priors = {int(s.name[1:]): np.array(json.loads((s / "train_prior.json").read_text())["effective_prior"])
              for s in sdirs if (s / "train_prior.json").exists()}
    return cfg, priors


def score_set(split: str, frames: list[pd.DataFrame], manifest: pd.DataFrame, classes: list[str],
              label_col: str, n_boot: int) -> list[dict]:
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
        # D2 / D9: share of images called malignant, and called each class (no true positives).
        fns = {"pred_malignant_rate": lambda y_, yb, P: float(M.pred_malignant(P, classes).mean())}
        for k, c in enumerate(classes):
            fns[f"pred_{c}_rate"] = lambda y_, yb, P, k=k: float((P.argmax(1) == k).mean())
    else:
        fns = M.metric_fns(classes, task)

    subsets = [("all", "all", np.ones(len(meta), bool))]
    for col in sub_cols:
        for val in sorted(meta[col].dropna().unique(), key=str):
            subsets.append((col, str(val), (meta[col] == val).to_numpy()))
    for col, val, mask in subsets:
        if mask.sum() < MIN_SUBGROUP:
            continue
        res = M.bootstrap(fns, y[mask], y_bin[mask], [P[mask] for P in Ps], groups[mask], n_boot=n_boot)
        for metric, r in res.items():
            rows.append({"eval_set": split, "subgroup_col": col, "subgroup": val, "metric": metric,
                         "mean": r["mean"], "std": r["std"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "n": int(mask.sum()), "n_malignant": int(np.nansum(y_bin[mask] == 1)),
                         "n_seeds": len(Ps)})
    return rows


def evaluate(exp: str, runs_dir="runs", results_dir="results", manifest_path=None,
             prior_mix: str | None = None, n_boot: int = 1000, splits=None,
             drop_diagnoses: list[str] | None = None, tag: str | None = None) -> pd.DataFrame:
    """drop_diagnoses: evaluation-time sensitivity check, e.g. without pyogenic granuloma;
    results go to results/<exp>+<tag>/."""
    exp_dir = Path(runs_dir) / exp
    cfg, priors = run_config(exp_dir)
    label_col = cfg.get("data", {}).get("label_col", "label_4class")
    classes = CLASSES[label_col]
    manifest = pd.read_csv(manifest_path or cfg.get("manifest", "data/manifest.csv"))
    name = exp if prior_mix is None else f"{exp}+prior_{prior_mix}"
    if drop_diagnoses:
        dropped = set(manifest.loc[manifest["diagnosis"].isin(drop_diagnoses), "image_id"])
        name += f"+{tag or 'drop'}"
    rows = []
    for split in splits or EVAL_SETS:
        seeds, frames = load_preds(exp_dir, split)
        if drop_diagnoses:
            frames = [f[~f["image_id"].isin(dropped)].reset_index(drop=True) for f in frames]
        if not frames or frames[0].empty:
            continue
        if prior_mix is not None:
            # D7: divide by the effective training prior, multiply by the Indian prior.
            cols = [f"p_{c}" for c in classes]
            for seed, f in zip(seeds, frames):
                pi = priors[seed]
                f[cols] = M.prior_shift(f[cols].to_numpy(), pi, indian_prior(classes, pi, prior_mix))
        rows += score_set(split, frames, manifest, classes, label_col, n_boot)
    out = pd.DataFrame(rows)
    out.insert(0, "exp", name)
    odir = Path(results_dir) / name
    odir.mkdir(parents=True, exist_ok=True)
    out.to_csv(odir / "metrics.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, nargs="+")
    ap.add_argument("--prior-mix", choices=[*INDIAN_MIXES, "all"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--drop-diagnoses", nargs="+", help="sensitivity check, e.g. 'pyogenic granuloma'")
    ap.add_argument("--tag", help="suffix for --drop-diagnoses results")
    args = ap.parse_args(argv)
    mixes = list(INDIAN_MIXES) if args.prior_mix == "all" else [args.prior_mix]
    for exp in args.exp:
        for mix in mixes:
            out = evaluate(exp, args.runs_dir, args.results_dir, prior_mix=mix, n_boot=args.n_boot,
                           drop_diagnoses=args.drop_diagnoses, tag=args.tag)
            head = out[(out["subgroup"] == "all") & out["metric"].isin(["macro_auc", "binary_auc", "missed_cancer_rate"])]
            print(head[["exp", "eval_set", "metric", "mean", "ci_lo", "ci_hi", "n_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
