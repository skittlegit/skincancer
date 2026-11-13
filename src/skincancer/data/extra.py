"""Training-only extra data, kept outside the frozen manifest.

    python -m skincancer.data.extra scin

Writes data/extra/<name>.csv in the manifest's columns with split = "train_extra". The
manifest, the groups and every validation / test split stay untouched, so models trained
with and without the extra data are scored on exactly the same images. A config opts in
with `data.extra_train: [scin]`. Extra images that near-duplicate any validation or test
image (same thresholds as data/dedup.py) are dropped.

Derm7pt (Kawahara et al., IEEE JBHI 2019; CC BY-NC-ND 4.0, never shown in figures): one
clinical (non-dermoscopic) photo per case from an Italian dermoscopy atlas. Mostly nevi and
melanomas, some BCC, no SCC. It records no skin type.

SCIN (Ward et al., JAMA Netw Open 2024; images contributed by the public through Google
Search): phone photos of everyday skin conditions with a dermatologists' differential and
no biopsy. Used as not cancer only: one photo per case, and a case is left out if any
cancer, pre-cancer or uncertain growth appears anywhere in its differential.
    python -m skincancer.data.extra scin --download    # fetch the tables and photos first
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_data_config
from ..labels import CANCER_TYPES, TUMOUR, map_diagnosis
from .dedup import cosine_pairs, embed, hash_pairs, phash_ints

EXTRA_DIR = Path("data/extra")
EXTRA_SPLIT = "train_extra"
NOT_EVALUATION = {"train", "excluded"}   # duplicates of these cannot leak into a score


def derm7pt_rows(root: Path) -> pd.DataFrame:
    base = next(p for p in (root / "derm7pt").rglob("meta.csv")).parent.parent   # release_v0/
    meta = pd.read_csv(base / "meta" / "meta.csv")
    rel = base.relative_to(root)
    label = meta["diagnosis"].map(lambda d: map_diagnosis(d)[0])
    ids = "derm7pt_" + meta["case_num"].astype(str)
    df = pd.DataFrame({
        "image_id": ids, "group_id": ids, "dataset": "derm7pt",
        "path": [(rel / "images" / p).as_posix() for p in meta["clinic"]],
        "diagnosis": meta["diagnosis"], "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), "unlabelled"),
    })
    # vascular lesions, melanosis and "miscellaneous" have no tumour label here: left out
    df["label_binary"] = label.map(lambda l: 1.0 if l in CANCER_TYPES else 0.0 if l == "benign" else np.nan)
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = meta.get("location")
    df["is_acral"] = meta.get("location", pd.Series("", index=meta.index)).astype(str).str.contains(
        "acral|palm|sole", case=False)
    df["patient_id"] = ids
    return df


SCIN_URL = "https://storage.googleapis.com/dx-scin-public-data/"
SCIN_NOT_SAFE = re.compile(
    "carcinoma|scc|melanoma|cancer|lymphoma|sarcoma|metasta|leukemia|paget|neoplasm|atypical nevus|"
    "actinic keratosis|lymphomatoid|arsenical|porokeratosis|histiocytosis|keratoacanthoma|bowen|"
    "mycosis fungoides|remove from labeling", re.I)


def scin_table(root: Path) -> pd.DataFrame:
    d = root / "scin"
    cases = pd.read_csv(d / "scin_cases.csv", dtype={"case_id": str})
    labels = pd.read_csv(d / "scin_labels.csv", dtype={"case_id": str})
    t = cases.merge(labels, on="case_id")
    t["conditions"] = t["weighted_skin_condition_label"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})
    return t


def scin_rows(root: Path) -> pd.DataFrame:
    t = scin_table(root)
    top = t["conditions"].map(lambda c: max(c, key=c.get) if c else "")
    safe = t["conditions"].map(lambda c: bool(c) and not any(SCIN_NOT_SAFE.search(k) for k in c))
    ids = "scin_" + t["case_id"].str.lstrip("-")
    df = pd.DataFrame({
        "image_id": ids, "group_id": ids, "dataset": "scin",
        "path": "scin/images/" + t["image_1_path"].astype(str).str.rsplit("/", n=1).str[-1],
        "diagnosis": top, "label_4class": "exclude",
        "label_fine": np.where(safe, "non_tumour", "unlabelled"),
    })
    df["label_binary"] = np.where(safe, 0.0, np.nan)   # ungraded, or a cancer somewhere in the differential
    df["label_cancer"] = df["label_binary"].map({0.0: "not_cancer"})
    df["fst"] = np.nan
    df["fst_bucket"] = np.nan
    df["site"] = np.nan
    df["is_acral"] = False
    df["patient_id"] = ids
    return df


def scin_download(root: Path, workers: int = 16) -> None:
    """Fetch the two tables, then the first photo of every usable case (about 1.2 MB each)."""
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from urllib.request import urlretrieve
    socket.setdefaulttimeout(60)   # a stalled connection fails and is fetched by the next run
    d = root / "scin"
    (d / "images").mkdir(parents=True, exist_ok=True)
    for f in ("scin_cases.csv", "scin_labels.csv"):
        if not (d / f).exists():
            urlretrieve(SCIN_URL + "dataset/" + f, d / f)
    rows = scin_rows(root)
    todo = [p for p in rows.loc[rows["label_binary"].notna(), "path"] if not (root / p).exists()]
    print(f"{len(todo)} photos to fetch", flush=True)

    def get(p: str) -> bool:
        tmp = root / (p + ".part")
        try:
            urlretrieve(SCIN_URL + "dataset/images/" + p.rsplit("/", 1)[-1], tmp)
            tmp.replace(root / p)
            return True
        except Exception as e:   # reported below; a second run fetches what is missing
            print("failed", p, e, flush=True)
            return False

    with ThreadPoolExecutor(workers) as ex:
        done = 0
        for i, ok in enumerate(ex.map(get, todo), 1):
            done += ok
            if i % 200 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} tried, {done} fetched", flush=True)


def drop_eval_duplicates(extra: pd.DataFrame, manifest: pd.DataFrame, root: Path, cfg: dict) -> pd.DataFrame:
    ev = manifest[~manifest["split"].isin(NOT_EVALUATION)].reset_index(drop=True)
    paths = [root / p for p in extra["path"]] + [root / p for p in ev["path"]]
    n = len(extra)
    dup = np.zeros(n, bool)
    for i, j in hash_pairs(phash_ints(paths), cfg["dedup"]["phash_max_distance"]):
        if i < n <= j:
            dup[i] = True
    for i, j in cosine_pairs(embed(paths), cfg["dedup"]["embed_min_cosine"]):
        if i < n <= j:
            dup[i] = True
    print(f"{dup.sum()} of {n} extra images near-duplicate one of {len(ev)} validation / test images: dropped")
    extra = extra.copy()
    extra.loc[dup, "exclude_reason"] = "dup_of_eval"
    return extra


def build(name: str, cfg: dict, check_duplicates: bool = True) -> pd.DataFrame:
    root = Path(cfg["data_root"])
    df = {"derm7pt": derm7pt_rows, "scin": scin_rows}[name](root)
    df["exclude_reason"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[df["label_binary"].isna(), "exclude_reason"] = "no_cancer_label"
    df.loc[~df["path"].map(lambda p: (root / p).exists()), "exclude_reason"] = "missing_image"
    keep = df["exclude_reason"].isna()
    if check_duplicates:
        checked = drop_eval_duplicates(df[keep].reset_index(drop=True), pd.read_csv(cfg["manifest"], low_memory=False),
                                       root, cfg)
        df = pd.concat([checked, df[~keep]], ignore_index=True)
    df["split"] = np.where(df["exclude_reason"].isna(), EXTRA_SPLIT, "excluded")
    return df


def load_extra(names: list[str], label_col: str, classes: list[str], extra_dir: Path = EXTRA_DIR) -> pd.DataFrame:
    """Rows of the named extra sets that may be trained on."""
    df = pd.concat([pd.read_csv(Path(extra_dir) / f"{n}.csv") for n in names], ignore_index=True)
    return df[(df["split"] == EXTRA_SPLIT) & df[label_col].isin(classes)].reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["derm7pt", "scin"])
    ap.add_argument("--download", action="store_true", help="scin only: fetch the tables and photos, then stop")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    if args.download:
        return scin_download(Path(cfg["data_root"]))
    df = build(args.name, cfg)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA_DIR / f"{args.name}.csv", index=False)
    print(pd.crosstab(df["split"], df["label_fine"], margins=True).to_string())
    print(df["exclude_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
