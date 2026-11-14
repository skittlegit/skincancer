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

MILK10k (ISIC collection 425, MILK study team, CC BY-NC 4.0; idea Q): clinical close-ups of
biopsied lesions from five centres. Only histopathology-confirmed lesions are used.
    python -m skincancer.data.extra milk10k --download && python -m skincancer.data.extra milk10k
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


MILK_API = "https://api.isic-archive.com/api/v2/images/search/?collections=425&limit=100"   # ISIC collection 425


def milk10k_download(root: Path, workers: int = 16) -> None:
    """Fetch the metadata of every MILK10k image, then the clinical close-ups (about 5,240)."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    import socket
    from urllib.request import urlopen, urlretrieve
    socket.setdefaulttimeout(60)
    d = root / "milk10k"
    (d / "images").mkdir(parents=True, exist_ok=True)
    meta = d / "metadata.json"
    if not meta.exists():
        rows, url = [], MILK_API
        while url:
            with urlopen(url) as r:
                page = json.load(r)
            rows += page["results"]
            url = page.get("next")
        meta.write_text(json.dumps(rows), encoding="utf-8")
    rows = json.loads(meta.read_text(encoding="utf-8"))
    todo = [(r["files"]["full"]["url"], d / "images" / f"{r['isic_id']}.jpg") for r in rows
            if r["metadata"]["acquisition"].get("image_type") == "clinical: close-up"
            and not (d / "images" / f"{r['isic_id']}.jpg").exists()]
    print(f"{len(todo)} clinical close-ups to fetch", flush=True)

    def get(job) -> bool:
        url, dst = job
        tmp = dst.with_suffix(".part")
        try:
            urlretrieve(url, tmp)
            tmp.replace(dst)
            return True
        except Exception as e:   # reported; a second run fetches what is missing
            print("failed", dst.name, e, flush=True)
            return False

    with ThreadPoolExecutor(workers) as ex:
        done = 0
        for i, ok in enumerate(ex.map(get, todo), 1):
            done += ok
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} tried, {done} fetched", flush=True)


def milk10k_rows(root: Path) -> pd.DataFrame:
    """MILK10k (ISIC collection 425; CC BY-NC 4.0, "MILK study team"): one clinical close-up
    per lesion from five centres, 95.7% of lesions biopsied or excised. Cancer when ISIC-DX
    says malignant and the most specific diagnosis maps to a cancer type; not cancer when
    ISIC-DX says benign; indeterminate lesions and actinic keratoses (never trained on, as in
    the manifest) are left out."""
    import json
    rows = json.loads((root / "milk10k" / "metadata.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if r["metadata"]["acquisition"].get("image_type") == "clinical: close-up"]
    clin = pd.DataFrame([r["metadata"]["clinical"] for r in rows])
    ids = pd.Series(["milk10k_" + r["isic_id"] for r in rows])
    specific = clin[[c for c in ("diagnosis_5", "diagnosis_4", "diagnosis_3", "diagnosis_2") if c in clin]].bfill(axis=1).iloc[:, 0]
    label = specific.map(lambda s: map_diagnosis(s)[0])
    top = clin["diagnosis_1"]
    biopsied = clin["diagnosis_confirm_type"] == "histopathology"   # 224 rest on one clinician's opinion: left out
    binary = np.where(biopsied & (top == "Malignant") & label.isin(CANCER_TYPES), 1.0,
                      np.where(biopsied & (top == "Benign") & (label != "ak"), 0.0, np.nan))
    df = pd.DataFrame({
        "image_id": ids, "group_id": "milk10k_" + clin["lesion_id"].astype(str), "dataset": "milk10k",
        "path": ["milk10k/images/" + r["isic_id"] + ".jpg" for r in rows],
        "diagnosis": specific, "label_4class": label,
        "label_fine": label.where(label.isin(TUMOUR), np.where(binary == 0.0, "other_benign_tumour", "unlabelled")),
    })
    df["label_binary"] = binary
    df["label_cancer"] = df["label_binary"].map({1.0: "cancer", 0.0: "not_cancer"})
    df["fst"] = np.nan        # not recorded
    df["fst_bucket"] = np.nan
    df["site"] = clin.get("anatom_site_1")
    df["is_acral"] = df["site"].astype(str).str.contains("acral|palm|sole", case=False)
    df["patient_id"] = df["group_id"]   # no patient ids are published: one photo per lesion
    return df


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
    df = {"derm7pt": derm7pt_rows, "scin": scin_rows, "milk10k": milk10k_rows}[name](root)
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
    ap.add_argument("name", choices=["derm7pt", "scin", "milk10k"])
    ap.add_argument("--download", action="store_true", help="scin, milk10k: fetch the tables and photos, then stop")
    ap.add_argument("--data-config", default="configs/data.yaml")
    args = ap.parse_args(argv)
    cfg = load_data_config(args.data_config)
    if args.download:
        return {"scin": scin_download, "milk10k": milk10k_download}[args.name](Path(cfg["data_root"]))
    df = build(args.name, cfg)
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA_DIR / f"{args.name}.csv", index=False)
    print(pd.crosstab(df["split"], df["label_fine"], margins=True).to_string())
    print(df["exclude_reason"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
