"""Lookalike confusion analysis (D9): which benign diagnoses get called which cancer, and
how often each cancer is called benign, per FST bucket; plus the highest-confidence errors.

    python -m skincancer.analysis.lookalike --exp e1_effb0 --split target_test

Writes results/<exp>/lookalike_<split>.csv and errors_<split>.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import load_preds, run_config
from ..labels import CLASSES


def analyse(exp: str, split: str, runs_dir="runs", results_dir="results", top_k: int = 10):
    exp_dir = Path(runs_dir) / exp
    cfg, _ = run_config(exp_dir)
    label_col = cfg["data"]["label_col"]
    classes = CLASSES[label_col]
    _, frames = load_preds(exp_dir, split)
    ids = frames[0]["image_id"]
    P = np.mean([f.set_index("image_id").loc[ids, [f"p_{c}" for c in classes]].to_numpy() for f in frames], 0)
    meta = pd.read_csv(cfg["manifest"]).set_index("image_id").loc[ids].reset_index()
    meta["pred"] = np.array(classes)[P.argmax(1)]
    meta["confidence"] = P.max(1)
    meta["true"] = meta[label_col]
    known = meta[meta["true"].isin(classes)]
    table = (known.groupby(["fst_bucket", "true", "diagnosis", "pred"], dropna=False).size()
             .rename("n").reset_index())
    table["share_of_diagnosis"] = table["n"] / table.groupby(["fst_bucket", "true", "diagnosis"],
                                                              dropna=False)["n"].transform("sum")
    errors = known[known["true"] != known["pred"]].sort_values("confidence", ascending=False)
    errors = errors.groupby(["true", "pred"]).head(top_k)[
        ["image_id", "path", "dataset", "fst_bucket", "diagnosis", "true", "pred", "confidence"]]
    out = Path(results_dir) / exp
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"lookalike_{split}.csv", index=False)
    errors.to_csv(out / f"errors_{split}.csv", index=False)
    return table, errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="target_test")
    args = ap.parse_args(argv)
    table, errors = analyse(args.exp, args.split)
    benign_called_cancer = table[(table["true"] == "benign") & (table["pred"] != "benign")]
    print(benign_called_cancer.sort_values("n", ascending=False).head(20).to_string(index=False))
    print(f"\n{len(errors)} example errors written")


if __name__ == "__main__":
    main()
