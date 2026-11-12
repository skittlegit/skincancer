"""Count check: images per split x class, per skin-type bucket, and why images were excluded.

    python -m skincancer.data.count_check
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import load_data_config


def report(m: pd.DataFrame) -> str:
    used = m[m["split"].fillna("excluded") != "excluded"]
    main = used[used["split"].isin(["train", "val", "test"])]
    lines = ["## Images per split x class",
             pd.crosstab(used["split"], used["label_5class"], margins=True).to_string(),
             "\n## train / val / test: split x skin-type bucket x class",
             pd.crosstab([main["split"], main["fst_bucket"]], main["label_5class"], margins=True).to_string(),
             "\n## train / val / test: dataset x class",
             pd.crosstab(main["dataset"], main["label_5class"], margins=True).to_string(),
             "\n## Exclusions", m["exclude_reason"].value_counts().to_string()]
    acral = used[(used["is_acral"] == True) & (used["label_4class"] == "mel")]  # noqa: E712
    lines.append(f"\nAcral melanoma images (approximate): {len(acral)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    print(report(pd.read_csv(cfg["manifest"], low_memory=False)))


if __name__ == "__main__":
    main()
