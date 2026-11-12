"""Count check: images per split x class and x type of lesion, and why images were excluded.

    python -m skincancer.data.count_check
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import load_data_config


def report(m: pd.DataFrame) -> str:
    used = m[m["split"].fillna("excluded") != "excluded"]
    main = used[used["split"].isin(["train", "val", "test"])]
    lines = ["## Images per split: cancer / not cancer",
             pd.crosstab(used["split"], used["label_cancer"].fillna("no label"), margins=True).to_string(),
             "\n## Images per split x type of lesion",
             pd.crosstab(used["split"], used["label_fine"], margins=True).to_string(),
             "\n## train / val / test: dataset x class",
             pd.crosstab([main["split"], main["dataset"]], main["label_cancer"], margins=True).to_string(),
             "\n## Exclusions", m["exclude_reason"].value_counts().to_string()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    print(report(pd.read_csv(cfg["manifest"], low_memory=False)))


if __name__ == "__main__":
    main()
