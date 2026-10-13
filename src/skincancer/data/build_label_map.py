"""Write data/label_map.csv: every (dataset, diagnosis) with its rule-based label (D5).

The team reviews and commits the file. Re-running only appends diagnoses that are not
already there, so manual edits survive; --force regenerates from scratch.

    python -m skincancer.data.build_label_map
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_data_config
from ..labels import map_diagnosis
from .sources import load_all


def build(meta: pd.DataFrame) -> pd.DataFrame:
    grp = meta.groupby(["dataset", "diagnosis"], dropna=False)
    rows = []
    for (dataset, diagnosis), g in grp:
        label, review, rule = map_diagnosis(diagnosis)
        partition = str(g["partition"].dropna().iloc[0]) if g["partition"].notna().any() else ""
        if partition == "non-neoplastic" and label != "exclude":
            # Fitzpatrick17k's own partition overrides keyword matches.
            label, review, rule = "exclude", True, "partition=non-neoplastic"
        if dataset == "ddi" and g["malignant_flag"].notna().any():
            ddi_malignant = bool(g["malignant_flag"].max())
            if ddi_malignant != (label in {"bcc", "scc", "mel"}) and label not in {"exclude_rare_cancer", "ak"}:
                review = True  # rule disagrees with DDI's biopsy flag
        rows.append({"dataset": dataset, "diagnosis": diagnosis, "n_images": len(g),
                     "partition": partition, "label_4class": label,
                     "needs_review": review, "rule": rule, "reviewed": False})
    return pd.DataFrame(rows).sort_values(["dataset", "label_4class", "n_images"],
                                          ascending=[True, True, False])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    out = Path(cfg["label_map"])

    fresh = build(load_all(cfg))
    if out.exists() and not args.force:
        old = pd.read_csv(out)
        key = ["dataset", "diagnosis"]
        new = fresh.merge(old[key], on=key, how="left", indicator=True)
        new = new[new["_merge"] == "left_only"].drop(columns="_merge")
        merged = pd.concat([old, new], ignore_index=True)
        print(f"{len(new)} new diagnoses appended to {out}")
    else:
        merged = fresh
        print(f"wrote {len(merged)} diagnoses to {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    summary = merged.groupby(["dataset", "label_4class"])["n_images"].sum().unstack(fill_value=0)
    print(summary.to_string())
    print(f"\n{int(merged['needs_review'].sum())} rows flagged needs_review. Check every row, "
          "set reviewed=True, and commit.")


if __name__ == "__main__":
    main()
