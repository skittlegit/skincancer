"""Build data/manifest.csv from the raw metadata and the committed label map.

    python -m skincancer.data.build_manifest

Columns: image_id, group_id, dataset, path, diagnosis, label_4class, label_fine,
label_cancer, label_binary, fst, fst_bucket, site, is_acral, patient_id, exclude_reason, split.
label_cancer (cancer / not_cancer) is what the model is trained on; label_binary is the same
as 1 / 0. label_4class is the label map's tumour label and label_fine a finer description;
both only describe results. `split` is left empty here; make_splits fills it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import CANCER_TYPES
from .sources import load_all

TRAIN_DATASETS = {"fitzpatrick17k", "pad_ufes_20"}


def attach_labels(meta: pd.DataFrame, label_map: pd.DataFrame) -> pd.DataFrame:
    lm = label_map[["dataset", "diagnosis", "label_4class"]]
    df = meta.merge(lm, on=["dataset", "diagnosis"], how="left")
    missing = df[df["label_4class"].isna()][["dataset", "diagnosis"]].drop_duplicates()
    if len(missing):
        raise ValueError(f"{len(missing)} diagnoses missing from the label map; run "
                         f"build_label_map and review:\n{missing.head(20).to_string()}")
    # Finer description. Only Fitzpatrick17k says which of its unmapped diagnoses are tumours
    # (cysts, syringoma, epidermal nevi ...) and which are not (psoriasis, eczema, acne ...).
    fitz = (df["dataset"] == "fitzpatrick17k") & (df["label_4class"] == "exclude")
    fine = df["label_4class"].replace({"exclude_rare_cancer": "rare_cancer", "exclude": "unlabelled"})
    fine = fine.mask(fitz & (df["partition"] == "non-neoplastic"), "non_tumour")
    fine = fine.mask(fitz & (df["partition"] == "benign"), "other_benign_tumour")
    df["label_fine"] = fine
    # Cancer or not. Actinic keratosis is precancerous: neither, so it gets no label.
    binary = fine.map(lambda f: 1.0 if f in ("bcc", "scc", "mel", "rare_cancer")
                      else 0.0 if f in ("benign", "non_tumour", "other_benign_tumour") else np.nan)
    assert CANCER_TYPES == {"bcc", "scc", "mel", "exclude_rare_cancer"}
    # DDI: the biopsy flag is the truth for every image.
    ddi = df["dataset"] == "ddi"
    df["label_binary"] = np.where(ddi, df["malignant_flag"], binary)
    # DermaCon-IN everyday clinic conditions (not lookalikes, not rare cancers): not cancer.
    other = (df["dataset"] == "dermacon_in") & (df["label_4class"] == "exclude")
    df.loc[other, "label_binary"] = 0.0
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    return df


def exclusion_reason(row, root: Path) -> str | float:
    if not (root / row["path"]).exists():
        return "missing_image"
    if row["dataset"] == "fitzpatrick17k" and isinstance(row["qc"], str) and row["qc"].strip().startswith("3"):
        return "qc_wrong_label"                # Fitzpatrick17k QC: "3 Wrongly labelled"
    if row["dataset"] == "dermacon_in" and isinstance(row["qc"], str) and row["qc"].strip().lower().startswith("no"):
        return "quality_insufficient"
    if pd.isna(row["label_binary"]) and row["label_4class"] != "ak":
        return "no_cancer_label"               # e.g. MIDAS "benign-other", "malignant-other"
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
        df["group_id"] = pd.Series(np.nan, index=df.index, dtype=object)   # filled with string ids below
    df["group_id"] = df["group_id"].fillna(df["patient_id"]).fillna(df["image_id"])
    df["split"] = ""
    cols = ["image_id", "group_id", "dataset", "path", "diagnosis", "label_4class",
            "label_fine", "label_cancer", "label_binary", "fst", "fst_bucket", "site", "is_acral",
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
