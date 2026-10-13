"""Domain probe: can a linear model tell source from target from frozen embeddings?
High accuracy means the features encode the domain; it should fall after CORAL.

    python -m skincancer.analysis.probe --run runs/e1_effb0/s0 runs/e7_coral_l1.0/s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import predict


def probe(run: str, a: str = "source_test", b: str = "target_test", seed: int = 0) -> dict:
    rdir = Path(run)
    ids_a, Fa = predict.embeddings(rdir, a)
    ids_b, Fb = predict.embeddings(rdir, b)
    rng = np.random.default_rng(seed)
    n = min(len(Fa), len(Fb))  # balanced, so chance = 50%
    ia, ib = rng.choice(len(Fa), n, replace=False), rng.choice(len(Fb), n, replace=False)
    X = np.vstack([Fa[ia], Fb[ib]])
    y = np.r_[np.zeros(n), np.ones(n)]
    cfg = yaml.safe_load((rdir / "config.yaml").read_text(encoding="utf-8"))
    group_of = pd.read_csv(cfg["manifest"]).set_index("image_id")["group_id"]
    groups = group_of.loc[np.r_[ids_a[ia], ids_b[ib]]].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    acc = cross_val_score(clf, X, y, cv=GroupKFold(5), groups=groups, scoring="accuracy")
    return {"run": run, "a": a, "b": b, "n_per_domain": int(n),
            "acc_mean": float(acc.mean()), "acc_std": float(acc.std())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--a", default="source_test")
    ap.add_argument("--b", default="target_test")
    args = ap.parse_args(argv)
    print(pd.DataFrame([probe(r, args.a, args.b) for r in args.run]).to_string(index=False))


if __name__ == "__main__":
    main()
