"""Download Fitzpatrick17k images from the URLs in its CSV, with a Wayback Machine fallback.

Get fitzpatrick17k.csv from https://github.com/mattgroh/fitzpatrick17k and put it at
data/raw/fitzpatrick17k/fitzpatrick17k.csv, then:

    python -m skincancer.data.download_fitzpatrick17k --neoplastic-only

The DermaAmin links (~3,400 neoplastic images) are dead. Their 2017-2021 snapshots on the
Wayback Machine are used instead. Each file's MD5 must equal the CSV's md5hash, so every
image saved is byte-identical to the original. Images are saved as <md5hash>.jpg; anything
that can't be recovered is logged to download_failures.csv. Safe to re-run: verified files
are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from ..config import load_data_config

HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset download; fitzpatrick17k)"}
WAYBACK = "https://web.archive.org/web/2021id_/{url}"          # id_ = raw bytes, closest snapshot
WAYBACK_API = "https://archive.org/wayback/available?url={url}"
DEAD_HOSTS = ("dermaamin.com",)  # every link 404s since ~2024; go straight to the archive
CDX_QUERY = ("https://web.archive.org/cdx/search/cdx?url=dermaamin.com/site/images/clinical-pic/"
             "&matchType=prefix&filter=statuscode:200&collapse=urlkey&fl=original,timestamp,digest,mimetype")
CDX: dict[str, tuple[str, str]] | None = None  # url key -> (archived original, timestamp)


def _key(url: str) -> str:
    return url.split("://", 1)[-1].replace("www.", "", 1).lower()


def load_cdx(path: Path) -> dict[str, tuple[str, str]]:
    """One listing of every archived DermaAmin image (~7k lines), cached on disk."""
    if not path.exists():
        data, err = _get(CDX_QUERY, timeout=300)
        if data is None:
            raise RuntimeError(f"CDX query failed: {err}")
        path.write_bytes(data)
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split(" ")
        if len(parts) >= 2:
            out.setdefault(_key(parts[0]), (parts[0], parts[1]))
    return out


class RateLimit:
    """Minimum interval between archive.org requests, shared by all threads: archive.org
    refuses connections for a while when hit too fast."""

    def __init__(self, interval: float):
        self.interval, self.lock, self.last = interval, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


ARCHIVE_LIMIT = RateLimit(1.5)


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _get(url: str, retries: int = 4, timeout: int = 60) -> tuple[bytes | None, str]:
    err = "unknown"
    for attempt in range(retries):
        if "archive.org" in url:
            ARCHIVE_LIMIT.wait()
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content, ""
            err = f"http {r.status_code}"
            if r.status_code in (404, 410):
                return None, err
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))   # archive.org rate limit: back off
        except requests.RequestException as e:
            err = type(e).__name__
            if isinstance(e, requests.ConnectionError) and "archive.org" in url:
                time.sleep(60 * (attempt + 1))   # blocked: wait it out instead of hammering
                continue
        time.sleep(2 * (attempt + 1))
    return None, err


def fetch(url: str, expected_md5: str, dest: Path, wayback: bool = True) -> str | None:
    """-> None on success, else an error string."""
    if dest.exists() and md5(dest.read_bytes()) == expected_md5:
        return None
    data, err = None, "dead host"
    if not any(h in url for h in DEAD_HOSTS):
        data, err = _get(url, retries=2)
        if data is not None and md5(data) == expected_md5:
            dest.write_bytes(data)
            return None
        if data is not None:
            err = "md5_mismatch_original"
    if not wayback:
        return err
    bare = url.split("://", 1)[-1]
    if CDX is not None:
        # Exact snapshot from the CDX index; URLs not in it were never archived.
        snap = CDX.get(_key(url))
        if snap is None:
            return f"{err}; wayback: not archived"
        data, werr = _get(f"https://web.archive.org/web/{snap[1]}id_/{snap[0]}")
        if data is not None and md5(data) == expected_md5:
            dest.write_bytes(data)
            return None
        # The indexed capture can be broken; try every other 200 capture of this URL.
        listing, _ = _get(f"https://web.archive.org/cdx/search/cdx?url={snap[0]}&filter=statuscode:200"
                          f"&fl=original,timestamp", retries=2)
        for line in (listing or b"").decode(errors="ignore").splitlines():
            parts = line.split(" ")
            if len(parts) < 2 or parts[1] == snap[1]:
                continue
            data, werr = _get(f"https://web.archive.org/web/{parts[1]}id_/{parts[0]}", retries=2)
            if data is not None and md5(data) == expected_md5:
                dest.write_bytes(data)
                return None
        return f"{err}; wayback: {'md5_mismatch' if data is not None else werr}"
    # No index: snapshots were taken over http:// but the CSV says https://, and playback
    # doesn't match across schemes, so try both.
    werr = ""
    for scheme in ("http://", "https://"):
        data, werr = _get(WAYBACK.format(url=scheme + bare))
        if data is not None and md5(data) == expected_md5:
            dest.write_bytes(data)
            return None
    # The default snapshot didn't match; ask for the closest one explicitly.
    meta, _ = _get(WAYBACK_API.format(url="http://" + bare), retries=2)
    if meta:
        import json
        snap = json.loads(meta).get("archived_snapshots", {}).get("closest", {})
        if snap.get("available"):
            raw = snap["url"].replace(f"/{snap['timestamp']}/", f"/{snap['timestamp']}id_/")
            data, werr = _get(raw)
            if data is not None and md5(data) == expected_md5:
                dest.write_bytes(data)
                return None
    return f"{err}; wayback: {'md5_mismatch' if data is not None else werr or 'no snapshot'}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--neoplastic-only", action="store_true",
                    help="skip images Fitzpatrick17k marks non-neoplastic (never used here)")
    ap.add_argument("--workers", type=int, default=3, help="keep low: archive.org rate-limits")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between archive.org requests")
    ap.add_argument("--no-wayback", action="store_true")
    args = ap.parse_args(argv)
    ARCHIVE_LIMIT.interval = args.interval
    cfg = load_data_config(args.data_config)
    ds = cfg["datasets"]["fitzpatrick17k"]
    root = Path(cfg["data_root"])
    df = pd.read_csv(root / ds["csv"])
    if args.neoplastic_only:
        df = df[df["three_partition_label"] != "non-neoplastic"]
    out_dir = root / ds["image_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["url"].notna()]
    if not args.no_wayback:
        global CDX
        CDX = load_cdx(root / "fitzpatrick17k" / "wayback_cdx.txt")
        print(f"Wayback index: {len(CDX)} archived DermaAmin images")

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch, row.url, row.md5hash, out_dir / f"{row.md5hash}.jpg", not args.no_wayback): row
                for row in df.itertuples()}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            err = fut.result()
            if err:
                row = futs[fut]
                failures.append({"md5hash": row.md5hash, "url": row.url, "error": err})
    pd.DataFrame(failures, columns=["md5hash", "url", "error"]).to_csv(
        root / "fitzpatrick17k" / "download_failures.csv", index=False)
    print(f"{len(df) - len(failures)} ok (md5-verified), {len(failures)} failed")


if __name__ == "__main__":
    main()
