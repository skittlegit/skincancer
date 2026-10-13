"""Build data/manifest.csv from the raw metadata and the committed label map.

    python -m skincancer.data.build_manifest

Columns: image_id, group_id, dataset, path, diagnosis, label_4class, label_3class,
label_binary, fst, fst_bucket, site, is_acral, patient_id, exclude_reason, split.
`split` is left empty here; make_splits fills it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import MALIGNANT, TO_3CLASS, TRAINABLE
from .sources import load_all

TRAIN_DATASETS = {"fitzpatrick17k", "pad_ufes_20"}


def attach_labels(meta: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    lm = label_map[["dataset", "diagnosis", "label_4class"]]
    df = meta.merge(lm, on=["dataset", "diagnosis"], how="left")
    missing = df[df["label_4class"].isna()][["dataset", "diagnosis"]].drop_duplicates()
    if len(missing):
        raise ValueError(f"{len(missing)} diagnoses missing from the label map; run "
                         f"build_label_map and review:\n{missing.head(20).to_string()}")
    df["label_3class"] = df["label_4class"].map(TO_3CLASS)
    binary = df["label_4class"].map(lambda l: 1.0 if l in MALIGNANT else (0.0 if l == "benign" else np.nan))
    # DDI: biopsy flag is the binary truth for every image (D9).
    ddi = df["dataset"] == "ddi"
    df["label_binary"] = np.where(ddi, df["malignant_flag"], binary)
    # DermaCon-IN non-cancer clinic images (not lookalikes, not rare cancers) -> binary 0 for
    # the dermacon_other false-alarm set (D9).
    other = (df["dataset"] == "dermacon_in") & (df["label_4class"] == "exclude")
    df.loc[other, "label_binary"] = 0.0
    return df


def exclusion_reason(row, root: Path) -> str | float:
    if not (root / row["path"]).exists():
        return "missing_image"
    if row["dataset"] in TRAIN_DATASETS:
        if isinstance(row["qc"], str) and row["qc"].strip().startswith("3"):
            return "qc_wrong_label"            # Fitzpatrick17k QC: "3 Wrongly labelled"
        if pd.isna(row["fst"]):
            return "unknown_fst"               # D4
    if row["dataset"] == "ddi":
        return np.nan                          # all DDI images are scored (binary)
    if row["dataset"] == "dermacon_in":
        if isinstance(row["qc"], str) and row["qc"].strip().lower().startswith("no"):
            return "quality_insufficient"
        if row["label_4class"] == "exclude":
            return np.nan                      # kept for dermacon_other (D9)
    if row["label_4class"] not in TRAINABLE and row["label_4class"] != "ak":
        return f"label_{row['label_4class']}"
    return np.nan


def build(cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    meta = load_all(cfg)
    df = attach_labels(meta, pd.read_csv(cfg["label_map"]))
    df["exclude_reason"] = df.apply(exclusion_reason, axis=1, root=root)
    groups_path = Path(cfg["groups"])
    if groups_path.exists():
        groups = pd.read_csv(groups_path)[["image_id", "group_id"]]
        df = df.merge(groups, on="image_id", how="left")
    else:
        df["group_id"] = np.nan
    df["group_id"] = df["group_id"].fillna(df["patient_id"]).fillna(df["image_id"])
    df["split"] = ""
    cols = ["image_id", "group_id", "dataset", "path", "diagnosis", "label_4class",
            "label_3class", "label_binary", "fst", "fst_bucket", "site", "is_acral",
            "patient_id", "exclude_reason", "split"]
    return df[cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    df = build(cfg)
    df.to_csv(cfg["manifest"], index=False)
    print(f"wrote {len(df)} rows to {cfg['manifest']}")
    print(df.groupby("dataset")["exclude_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
