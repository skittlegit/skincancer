"""Print docs/PROJECT_EXPLAINER.html to docs/PROJECT_EXPLAINER.pdf with headless Chrome or Edge.

    python scripts/export_explainer.py [--browser PATH]

Edit the HTML (numbers, date, the "Results: data vN" chip), then re-run. Figures are read
from reports/figures/, so rebuild them first with `python -m skincancer.report ...`.
Needs Chrome/Edge 131+ for the running header and page numbers (CSS @page margin boxes).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROJECT_EXPLAINER.html"
OUT = ROOT / "docs" / "PROJECT_EXPLAINER.pdf"
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def find_browser() -> str:
    for c in CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("No Chrome/Edge found; pass --browser PATH")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", default=None)
    args = ap.parse_args(argv)
    browser = args.browser or find_browser()
    with tempfile.TemporaryDirectory() as profile:   # own profile, so an open browser is left alone
        subprocess.run([
            browser, "--headless=new", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer",
            "--allow-file-access-from-files", "--virtual-time-budget=15000",   # lets web fonts load
            f"--user-data-dir={profile}", f"--print-to-pdf={OUT}", SRC.as_uri(),
        ], check=True, timeout=300)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
