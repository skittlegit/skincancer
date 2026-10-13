"""Download DermaCon-IN from Harvard Dataverse (doi:10.7910/DVN/W7OUZM, CC BY-NC-SA 4.0).

    python -m skincancer.data.download_dermacon_in

Fetches the metadata and both image zips. The zips are fetched in parallel byte ranges
(a single stream from Dataverse's S3 runs at ~50 KB/s), verified against Dataverse's
MD5s, and extracted into data/raw/dermacon_in/DATASET/.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/W7OUZM"
CHUNK = 32 * 1024 * 1024
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Mozilla/5.0 (research dataset download; dermacon-in)"  # default UA gets 403


def file_list() -> list[dict]:
    r = HTTP.get(f"{API}/datasets/:persistentId/", params={"persistentId": DOI}, timeout=60)
    r.raise_for_status()
    return [f["dataFile"] for f in r.json()["data"]["latestVersion"]["files"]]


def _signed_url(file_id: int) -> str:
    r = HTTP.get(f"{API}/access/datafile/{file_id}", allow_redirects=False, timeout=60)
    return r.headers.get("Location") or f"{API}/access/datafile/{file_id}"


def _get_range(file_id: int, start: int, end: int, part: Path) -> None:
    want = end - start + 1
    if part.exists() and part.stat().st_size == want:
        return
    for _ in range(5):
        try:
            r = HTTP.get(_signed_url(file_id), headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == want:
                part.write_bytes(r.content)
                return
        except requests.RequestException:
            pass
    raise RuntimeError(f"range {start}-{end} of file {file_id} failed")


def download(f: dict, out_dir: Path, workers: int) -> Path:
    dest = out_dir / f["filename"]
    if dest.exists() and dest.stat().st_size == f["filesize"]:
        return dest
    parts_dir = out_dir / f".parts_{f['id']}"
    parts_dir.mkdir(exist_ok=True)
    ranges = [(s, min(s + CHUNK, f["filesize"]) - 1) for s in range(0, f["filesize"], CHUNK)]
    with ThreadPoolExecutor(workers) as ex, tqdm(total=len(ranges), desc=f["filename"]) as bar:
        futs = [ex.submit(_get_range, f["id"], s, e, parts_dir / f"{i:05d}") for i, (s, e) in enumerate(ranges)]
        for fut in futs:
            fut.result()
            bar.update()
    h = hashlib.md5()
    with open(dest, "wb") as out:
        for i in range(len(ranges)):
            data = (parts_dir / f"{i:05d}").read_bytes()
            h.update(data)
            out.write(data)
    if h.hexdigest() != f["md5"]:
        dest.unlink()
        raise RuntimeError(f"{f['filename']}: md5 mismatch")
    for p in parts_dir.iterdir():
        p.unlink()
    parts_dir.rmdir()
    return dest


def extract_images(zip_path: Path, image_dir: Path) -> int:
    """Flatten image files into image_dir (basenames only, so no path traversal)."""
    image_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            dest = image_dir / name
            if not dest.exists() or dest.stat().st_size != info.file_size:
                dest.write_bytes(z.read(info))
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/dermacon_in")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {f["filename"]: f for f in file_list()}
    meta = files["Skin_Metadata.tab"]
    r = HTTP.get(f"{API}/access/datafile/{meta['id']}", params={"format": "original"}, timeout=120)
    r.raise_for_status()
    (out / "Skin_Metadata.csv").write_bytes(r.content)
    total = 0
    for name in ("DATASET_0.zip", "DATASET_1.zip"):
        z = download(files[name], out, args.workers)
        total += extract_images(z, out / "DATASET")
    print(f"{total} images in {out / 'DATASET'}")


if __name__ == "__main__":
    main()
