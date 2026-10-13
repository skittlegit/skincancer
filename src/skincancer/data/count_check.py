"""Week-2 count check: images per class x FST bucket x split, and the D6 merge rule.

    python -m skincancer.data.count_check
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import load_data_config

MIN_SCC_TARGET_TEST = 15  # D6


def report(m: pd.DataFrame) -> tuple[str, bool]:
    used = m[m["split"].fillna("excluded") != "excluded"]
    lines = ["## Images per split x class", pd.crosstab(used["split"], used["label_4class"], margins=True).to_string(),
             "\n## Trainable pools: class x FST bucket"]
    pools = used[used["split"].str.startswith(("source_", "target_"))]
    lines.append(pd.crosstab([pools["dataset"], pools["fst_bucket"]], pools["label_4class"], margins=True).to_string())
    lines.append("\n## Exclusions")
    lines.append(m["exclude_reason"].value_counts().to_string())
    acral = used[(used["is_acral"] == True) & (used["label_4class"] == "mel")]  # noqa: E712
    lines.append(f"\nAcral melanoma images (approximate, D9): {len(acral)}")
    scc = int(((used["split"] == "target_test") & (used["label_4class"] == "scc")).sum())
    merge = scc < MIN_SCC_TARGET_TEST
    lines.append(f"\nSCC images in target_test: {scc} (threshold {MIN_SCC_TARGET_TEST}) -> "
                 + ("MERGE: set data.label_col: label_3class in configs/base.yaml and record it in DECISIONS.md D6"
                    if merge else "keep 4 classes"))
    return "\n".join(lines), merge


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    text, _ = report(pd.read_csv(cfg["manifest"]))
    print(text)


if __name__ == "__main__":
    main()
