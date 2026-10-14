"""Download MRA-MIDAS clinical photos from Redivis (Stanford AIMI; access approval needed).

    python -m skincancer.data.download_midas

The first run opens a Redivis sign-in in the browser (OAuth device flow); credentials are
cached in ~/.redivis, never in the repo. Downloads the metadata table and the clinical
photos only (6 in and 1 ft; no dermoscopy or telehealth captures, to keep the modality
fixed), each verified against Redivis's MD5. Saved to data/raw/midas/ (git-ignored;
MIDAS is test-only, like DDI, and its images are never shown in figures).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATASET = "mra_midas_multimodal_image_dataset_for_ai_based_skin_cancer:dpx2"
CLINICAL = ("6in", "1ft")


def _stem(name) -> str:
    return Path(str(name)).stem.lower().replace("_cropped", "")


def match_files(meta: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Metadata row -> file in the index: exact file name first, then by stem."""
    exact = meta.merge(index, left_on="midas_file_name", right_on="file_name", how="left")
    miss = exact["file_id"].isna()
    idx = index.assign(stem=index["file_name"].map(_stem)).drop_duplicates("stem")
    by_stem = meta[miss.values].assign(stem=lambda d: d["midas_file_name"].map(_stem)).merge(
        idx, on="stem", how="left").drop(columns="stem")
    out = pd.concat([exact[~miss], by_stem], ignore_index=True)
    return out


def main(argv=None):
    import redivis

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/midas")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    ds = redivis.organization("AIMI").dataset(DATASET)
    meta = ds.table("release_midas").to_pandas_dataframe()
    meta.to_csv(out / "release_midas.csv", index=False)
    index = ds.table("data").to_pandas_dataframe()
    index.to_csv(out / "file_index.csv", index=False)

    rows = match_files(meta[meta["midas_distance"].isin(CLINICAL)], index)
    rows = rows[rows["file_id"].notna()]
    files_table = ds.table("data")
    errors: list[str] = []

    def get(r) -> str | None:
        dest = out / "images" / r.file_name
        want = base64.b64decode(r.md5_hash).hex()
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == want:
            return None
        for _ in range(3):
            try:
                data = files_table.file(r.file_name).read()
                if hashlib.md5(data).hexdigest() == want:
                    dest.write_bytes(data)
                    return None
                errors.append(f"{r.file_name}: md5 mismatch")
            except Exception as e:  # noqa: BLE001 - transient network errors: retry
                errors.append(f"{r.file_name}: {type(e).__name__}: {e}"[:300])
        return r.file_name

    with ThreadPoolExecutor(args.workers) as ex:
        failed = [f for f in tqdm(ex.map(get, rows.itertuples()), total=len(rows)) if f]
    rows[["midas_file_name", "file_name"]].to_csv(out / "clinical_files.csv", index=False)
    print(f"{len(rows) - len(failed)} clinical images ok (md5-verified), {len(failed)} failed")
    if errors:
        print("first errors:", *errors[:3], sep="\n  ")


if __name__ == "__main__":
    main()
