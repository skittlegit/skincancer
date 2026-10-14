"""Training-only extra data, kept outside the frozen manifest (D16).

    python -m skincancer.data.extra derm7pt

Writes data/extra/<name>.csv in the manifest's columns with split = "source_train_extra".
The manifest, groups and every validation / test split stay untouched; a config opts in
with `data.extra_train: [derm7pt]`. Images that near-duplicate any evaluation image
(pHash <= 2 or embedding cosine >= 0.90, D10) are dropped.

Derm7pt (Kawahara et al., IEEE JBHI 2019; CC BY-NC-ND 4.0): one clinical (non-dermoscopic)
photo per case from an Italian dermoscopy atlas. It has no Fitzpatrick type; the population
is taken as FST I-III (D16).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import TO_3CLASS, TRAINABLE, is_malignant, map_diagnosis
from .dedup import cosine_pairs, embed, hash_pairs, phash_ints

EXTRA_DIR = Path("data/extra")
TRAIN_SPLITS = {"source_train", "target_train"}   # duplicates of these cannot leak
DERM7PT_OVERRIDES = {"melanoma metastasis": "exclude"}   # not a primary skin lesion


def derm7pt_rows(root: Path) -> pd.DataFrame:
    base = next(p for p in (root / "derm7pt").rglob("meta.csv")).parent.parent   # release_v0/
    meta = pd.read_csv(base / "meta" / "meta.csv")
    rel = base.relative_to(root)
    labels = meta["diagnosis"].map(lambda d: DERM7PT_OVERRIDES.get(d) or map_diagnosis(d)[0])
    df = pd.DataFrame({
        "image_id": "derm7pt_" + meta["case_num"].astype(str),
        "dataset": "derm7pt",
        "path": [(rel / "images" / p).as_posix() for p in meta["clinic"]],
        "diagnosis": meta["diagnosis"],
        "fst": np.nan,
        "fst_bucket": "I-III (assumed)",
        "patient_id": "derm7pt_" + meta["case_num"].astype(str),
        "site": meta.get("location"),
        "is_acral": meta.get("location", pd.Series("", index=meta.index)).astype(str).str.contains(
            "acral|palm|sole", case=False),
        "label_4class": labels,
    })
    df["label_3class"] = df["label_4class"].map(TO_3CLASS)
    df["label_binary"] = df["label_4class"].map(is_malignant).astype(int)
    df["group_id"] = df["patient_id"]
    return df


def drop_eval_duplicates(extra: pd.DataFrame, manifest: pd.DataFrame, root: Path, cfg: dict) -> pd.DataFrame:
    ev = manifest[~manifest["split"].isin(TRAIN_SPLITS | {"excluded"})].reset_index(drop=True)
    paths = [root / p for p in extra["path"]] + [root / p for p in ev["path"]]
    n = len(extra)
    dup = np.zeros(n, bool)
    hashes = phash_ints(paths)
    for i, j in hash_pairs(hashes, cfg["dedup"]["phash_max_distance"]):
        if i < n <= j:
            dup[i] = True
    emb = embed(paths)
    for i, j in cosine_pairs(emb, cfg["dedup"]["embed_min_cosine"]):
        if i < n <= j:
            dup[i] = True
    print(f"{dup.sum()} of {n} extra images near-duplicate an evaluation image: dropped")
    extra = extra.copy()
    extra.loc[dup, "exclude_reason"] = "dup_of_eval"
    return extra


def build(name: str, cfg: dict) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    df = {"derm7pt": derm7pt_rows}[name](root)
    missing = ~df["path"].map(lambda p: (root / p).exists())
    df["exclude_reason"] = np.where(missing, "missing_image", None)
    df.loc[~missing & ~df["label_4class"].isin(TRAINABLE), "exclude_reason"] = "label_" + df["label_4class"]
    keep = df["exclude_reason"].isna()
    checked = drop_eval_duplicates(df[keep].reset_index(drop=True), pd.read_csv(cfg["manifest"]), root, cfg)
    df = pd.concat([checked, df[~keep]], ignore_index=True)
    df["split"] = np.where(df["exclude_reason"].isna(), "source_train_extra", "excluded")
    return df


def load_extra(names: list[str], label_col: str, classes: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(EXTRA_DIR / f"{n}.csv") for n in names]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["split"] == "source_train_extra") & df[label_col].isin(classes)]
    return df.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["derm7pt"])
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    df = build(args.name, cfg)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA_DIR / f"{args.name}.csv", index=False)
    print(pd.crosstab(df["split"], df["label_4class"], margins=True).to_string())
    print(df["exclude_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
