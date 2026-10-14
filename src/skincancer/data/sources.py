"""Loaders that turn each dataset's own metadata into one standard frame.

Standard columns: image_id, dataset, path (relative to data_root), diagnosis, fst,
fst_bucket, patient_id, site, is_acral, malignant_flag, partition, qc.
A dataset whose metadata file is missing is skipped with a warning, so the pipeline
runs before DDI / DermaCon-IN access arrives.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..labels import DDI_BINS, clean_fst, fst_bucket

COLUMNS = ["image_id", "dataset", "path", "diagnosis", "fst", "fst_bucket", "patient_id",
           "site", "is_acral", "malignant_flag", "partition", "qc"]


def _frame(**cols) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def _acral_from_diagnosis(diag: pd.Series) -> pd.Series:
    return diag.str.lower().str.contains("acral", na=False)


def load_fitzpatrick17k(ds: dict, root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / ds["csv"])
    fst = df["fitzpatrick_scale"].map(clean_fst)
    return _frame(
        image_id="f17k_" + df["md5hash"].astype(str),
        dataset="fitzpatrick17k",
        path=ds["image_dir"] + "/" + df["md5hash"].astype(str) + ".jpg",
        diagnosis=df["label"],
        fst=fst,
        fst_bucket=fst.map(fst_bucket),
        is_acral=_acral_from_diagnosis(df["label"]),
        partition=df.get("three_partition_label"),
        qc=df.get("qc"),
    )


def load_pad_ufes_20(ds: dict, root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / ds["csv"])
    fst = df["fitspatrick"].map(clean_fst)  # sic: the column is spelled this way
    region = df["region"].astype(str).str.upper()
    return _frame(
        image_id="pad_" + df["img_id"].astype(str).str.replace(r"\.png$", "", regex=True),
        dataset="pad_ufes_20",
        path=ds["image_dir"] + "/" + df["img_id"].astype(str),
        diagnosis=df["diagnostic"],
        fst=fst,
        fst_bucket=fst.map(fst_bucket),
        patient_id="pad_" + df["patient_id"].astype(str),
        site=region,
        is_acral=region.isin(["HAND", "FOOT"]),  # approximate: palm vs dorsum unknown (D9)
    )


def load_ddi(ds: dict, root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / ds["csv"])
    malignant = df["malignant"].astype(str).str.lower().isin(["true", "1"])
    return _frame(
        image_id="ddi_" + df["DDI_ID"].astype(str),
        dataset="ddi",
        path=ds["image_dir"] + "/" + df["DDI_file"].astype(str),
        diagnosis=df["disease"],
        fst_bucket=df["skin_tone"].map(DDI_BINS),
        patient_id="ddi_" + df["DDI_ID"].astype(str),
        is_acral=_acral_from_diagnosis(df["disease"]),
        malignant_flag=malignant.astype(float),
    )


def load_dermacon_in(ds: dict, root: Path) -> pd.DataFrame:
    col = ds["columns"]
    df = pd.read_csv(root / ds["csv"])
    fst = df[col["fst"]].map(clean_fst) if col.get("fst") in df else pd.Series(np.nan, index=df.index)
    pid = df[col["patient_id"]].astype(str) if col.get("patient_id") in df else df[col["image"]].astype(str)
    return _frame(
        image_id="dci_" + df[col["image"]].astype(str),
        dataset="dermacon_in",
        path=ds["image_dir"] + "/" + df[col["image"]].astype(str),
        diagnosis=df[col["diagnosis"]],
        fst=fst,
        fst_bucket=fst.map(fst_bucket),
        patient_id="dci_" + pid,
        site=df[col["site"]] if col.get("site") in df else np.nan,
        is_acral=_acral_from_diagnosis(df[col["diagnosis"]].astype(str)),
        qc=df[col["quality"]] if col.get("quality") in df else np.nan,
    )


ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
ACRAL_SITE = r"palm|sole|plantar|\bfoot\b|\bfeet\b|\bhand\b|finger|\btoe|nail|heel"


def load_midas(ds: dict, root: Path) -> pd.DataFrame:
    """MRA-MIDAS clinical photos (6 in / 1 ft). Diagnosis = pathology; unbiopsied control
    lesions have none and are excluded downstream. Two photos of one lesion share a group."""
    meta = pd.read_csv(root / ds["csv"])
    files = pd.read_csv(root / ds["files"])          # midas_file_name -> downloaded file_name
    df = meta.merge(files, on="midas_file_name", how="inner")
    fst = df["midas_fitzpatrick"].astype(str).str.extract(r"^\s*(i{1,3}|iv|vi?|v)\b", expand=False)
    fst = fst.map(ROMAN).astype(float)
    site = df["midas_location"].astype(str).str.lower()
    return _frame(
        image_id="midas_" + df["file_name"].astype(str).str.replace(r"\.[^.]+$", "", regex=True),
        dataset="midas",
        path=ds["image_dir"] + "/" + df["file_name"].astype(str),
        diagnosis=df["midas_path"].fillna("no pathology (control)"),
        fst=fst,
        fst_bucket=fst.map(fst_bucket),
        patient_id="midas_" + df["midas_record_id"].astype(str),
        site=site,
        is_acral=site.str.contains(ACRAL_SITE, na=False),
    )


LOADERS = {
    "fitzpatrick17k": load_fitzpatrick17k,
    "pad_ufes_20": load_pad_ufes_20,
    "ddi": load_ddi,
    "dermacon_in": load_dermacon_in,
    "midas": load_midas,
}


def load_all(data_cfg: dict) -> pd.DataFrame:
    root = Path(data_cfg["data_root"])
    frames = []
    for name, ds in data_cfg["datasets"].items():
        if not (root / ds["csv"]).exists():
            warnings.warn(f"{name}: {root / ds['csv']} not found, skipping")
            continue
        frames.append(LOADERS[name](ds, root))
    if not frames:
        raise FileNotFoundError(f"No dataset metadata found under {root}")
    return pd.concat(frames, ignore_index=True)
