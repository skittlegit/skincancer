"""Score a finished run on more splits, and cache embeddings.

    # test-only sets: once per finished method, recorded in runs/final_ledger.csv (D9)
    python -m skincancer.predict --run runs/e1_effb0/s0 --splits ddi_test dermacon_test pad_dark_test --final
    # embeddings for probe / UMAP / fusion
    python -m skincancer.predict --run runs/e1_effb0/s0 --splits source_test target_test --embeddings
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import runs

TEST_ONLY = {"ddi_test", "dermacon_test", "dermacon_other", "pad_dark_test", "midas_test"}
LEDGER = Path("runs/final_ledger.csv")


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def check_ledger(run: str, splits: list[str], force: bool, ledger: Path = LEDGER) -> None:
    if not ledger.exists() or force:
        return
    done = pd.read_csv(ledger)
    hit = done[(done["run"] == run) & (done["split"].isin(splits))]
    if len(hit):
        raise SystemExit(f"{run} was already scored on {sorted(hit['split'])} "
                         f"(see {ledger}). Test-only sets are scored once per method; --force overrides.")


def append_ledger(run: str, splits: list[str], ledger: Path = LEDGER) -> None:
    rows = pd.DataFrame({"run": run, "split": splits, "commit": _commit(),
                         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ledger, mode="a", header=not ledger.exists(), index=False)


def embeddings(rdir: Path, split: str, model=None, cfg=None, manifest=None):
    """-> (image_ids, features); cached at <run>/emb/<split>.npz."""
    path = rdir / "emb" / f"{split}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["image_id"], z["features"]
    if model is None:
        model, cfg = runs.load_run(rdir)
    manifest = manifest if manifest is not None else pd.read_csv(cfg["manifest"])
    df = runs.split_rows(manifest, split)
    _, F = runs.infer(model, runs.make_loader(df, cfg, train=False), with_features=True)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, image_id=df["image_id"].to_numpy(), features=F)
    return df["image_id"].to_numpy(), F


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--final", action="store_true", help="required for test-only sets")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    args = ap.parse_args(argv)
    rdir = Path(args.run)
    test_only = [s for s in args.splits if s in TEST_ONLY]
    if test_only and not args.final:
        raise SystemExit(f"{test_only} are test-only: pass --final (scored once per method, D9)")
    if args.final:
        check_ledger(str(rdir.as_posix()), test_only, args.force)
    model, cfg = runs.load_run(rdir)
    manifest = pd.read_csv(cfg["manifest"])
    classes = runs.classes_of(cfg)
    (rdir / "preds").mkdir(exist_ok=True)
    scored = []
    for split in args.splits:
        df = runs.split_rows(manifest, split)
        if len(df) == 0:
            print(f"{split}: no rows, skipped")
            continue
        P, F = runs.infer(model, runs.make_loader(df, cfg, train=False), with_features=True)
        runs.prediction_frame(df, P, classes, cfg).to_csv(rdir / "preds" / f"{split}.csv", index=False)
        if args.embeddings:
            (rdir / "emb").mkdir(exist_ok=True)
            np.savez(rdir / "emb" / f"{split}.npz", image_id=df["image_id"].to_numpy(), features=F)
        scored.append(split)
        print(f"{split}: {len(df)} images")
    if args.final:
        append_ledger(str(rdir.as_posix()), [s for s in scored if s in TEST_ONLY])


if __name__ == "__main__":
    main()
