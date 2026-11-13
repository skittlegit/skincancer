"""Make a new model out of finished runs without retraining. This is how checkpoint 2 is built
from checkpoint 1's weights (docs/CHECKPOINTS.md).

    # test-time flips: the same weights, predicting on the image and its mirror images
    python -m skincancer.derive tta --exp skinnet-c1 --name skinnet-c2-tta
    # seed average: one prediction per image, the mean of the seeds' probabilities
    python -m skincancer.derive ensemble --exp skinnet-c2-tta --name skinnet-c2 --splits val test

`tta` writes runs/<name>/s<seed>/ with the weights and a config that has eval.tta on; score it
with `predict` as usual. `ensemble` writes runs/<name>/s0/preds/<split>.csv from the source
runs' prediction files. Averaging predictions on a test-only set is a new look at that set:
it needs --final and is recorded in the ledger like any other.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .labels import CLASSES
from .predict import LEDGER, TEST_ONLY, append_ledger, check_ledger


def make_tta(exp: str, name: str, runs_dir: str = "runs") -> list[Path]:
    out = []
    for sdir in sorted((Path(runs_dir) / exp).glob("s*")):
        dst = Path(runs_dir) / name / sdir.name
        dst.mkdir(parents=True, exist_ok=True)
        cfg = yaml.safe_load((sdir / "config.yaml").read_text(encoding="utf-8"))
        cfg["name"] = name
        cfg["eval"]["tta"] = True
        (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        for f in ("best.pt", "summary.json", "history.csv"):
            if (sdir / f).exists():
                shutil.copy(sdir / f, dst / f)
        out.append(dst)
    return out


def make_ensemble(exp: str, name: str, splits: list[str], runs_dir: str = "runs", final: bool = False,
                  force: bool = False, ledger: Path = LEDGER) -> Path:
    """force: rebuild predictions on a test-only set that the ledger already lists for this
    model (a reproduction, not a new look); nothing is added to the ledger."""
    src = sorted((Path(runs_dir) / exp).glob("s*"))
    if not src:
        raise FileNotFoundError(f"no runs under {Path(runs_dir) / exp}")
    dst = Path(runs_dir) / name / "s0"
    test_only = [s for s in splits if s in TEST_ONLY]
    if test_only and not final:
        raise SystemExit(f"{test_only} are test-only: pass --final (scored once per model)")
    if final:
        check_ledger(dst.as_posix(), test_only, force, ledger)
    cfg = yaml.safe_load((src[0] / "config.yaml").read_text(encoding="utf-8"))
    cols = [f"p_{c}" for c in CLASSES[cfg["data"]["label_col"]]]
    cfg.update(name=name, seed=0, ensemble_of=[s.as_posix() for s in src])
    (dst / "preds").mkdir(parents=True, exist_ok=True)
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for split in splits:
        frames = [pd.read_csv(s / "preds" / f"{split}.csv") for s in src]
        ids = frames[0]["image_id"]
        for f in frames[1:]:
            assert set(f["image_id"]) == set(ids), f"{split}: seeds scored different images"
        P = np.mean([f.set_index("image_id").loc[ids, cols].to_numpy() for f in frames], axis=0)
        out = pd.DataFrame(P, columns=cols)
        out.insert(0, "image_id", ids.to_numpy())
        out.insert(1, "exp", name)
        out.insert(2, "seed", 0)
        out.to_csv(dst / "preds" / f"{split}.csv", index=False)
    if final and not force:
        append_ledger(dst.as_posix(), test_only, ledger)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["tta", "ensemble"])
    ap.add_argument("--exp", required=True, help="finished experiment to derive from")
    ap.add_argument("--name", required=True, help="name of the derived experiment")
    ap.add_argument("--splits", nargs="+", default=["val"], help="ensemble only")
    ap.add_argument("--final", action="store_true", help="ensemble only: required for test-only sets")
    ap.add_argument("--force", action="store_true", help="ensemble only: rebuild sets the ledger already lists")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)
    if args.what == "tta":
        for d in make_tta(args.exp, args.name, args.runs_dir):
            print(f"wrote {d}")
    else:
        print(f"wrote {make_ensemble(args.exp, args.name, args.splits, args.runs_dir, args.final, args.force)}")


if __name__ == "__main__":
    main()
