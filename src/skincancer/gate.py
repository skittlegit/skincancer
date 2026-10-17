"""E10 / E10b: a "not a tumour" gate in front of the Derm Foundation classifier (D26, D27).

The gate is a logistic regression on frozen Derm Foundation embeddings, tumour vs non-tumour.
The system multiplies the gate's P(tumour) into e1_fm_lr's cancer probabilities and is written as
a normal run, so evaluate.py scores it like any other experiment.

  e10   tumours = source_train (FST I–III); non-tumours = Fitzpatrick17k images outside the four
        classes, FST I–III only (data/gate/nontumour.csv); class-balanced; C on gate val AUC.
  e10b  adds dark skin to both sides: target_train tumours (FST IV–VI) and 1,000 FST IV–VI
        non-tumours (data/gate/nontumour_dark.csv). Weighting (class-balanced, or every
        class × skin group cell equal) and C are chosen on the mean of the light and dark gate val
        AUCs; t95 keeps 95% of val tumours in each skin group.

    python -m skincancer.gate [--variant e10b]
    python -m skincancer.gate [--variant e10b] --final ddi_test dermacon_test dermacon_other pad_dark_test midas_test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import runs
from .config import load_config
from .foundation import LR_C, LR_C_TARGET, load_embeddings, write_run

GATE_LIST = "data/gate/nontumour.csv"
BASE = "e1_fm_lr"
KEEP = 0.95          # the cut-off t95 keeps this share of val tumours (in each skin group for e10b)
VARIANTS = {
    "e10": dict(run="e10_gate", lists=[GATE_LIST], C=LR_C, weighting=["class"],
                tumours={"train": ["source_train"], "val": ["source_val"]}),
    "e10b": dict(run="e10b_gate", lists=[GATE_LIST, "data/gate/nontumour_dark.csv"], C=LR_C_TARGET,
                 weighting=["class", "cell"],
                 tumours={"train": ["source_train", "target_train"], "val": ["source_val", "target_val"]}),
}


def gated(P: np.ndarray, g: np.ndarray, classes: list[str]) -> np.ndarray:
    """p'(cancer) = g·p(cancer); p'(benign) = 1 − g·Σ p(cancer). Rows still sum to 1."""
    b = classes.index("benign")
    out = P * g[:, None]
    out[:, b] = 1.0 - (out.sum(1) - out[:, b])
    return out


def skin(fst_bucket: pd.Series) -> np.ndarray:
    return np.where(fst_bucket.isin(["IV", "V-VI"]), "dark", "light")


def cell_weights(y: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Sample weights giving every (class, skin group) cell the same total weight."""
    w = np.zeros(len(y))
    cells = {(c, s) for c, s in zip(y, grp)}
    for c, s in cells:
        m = (y == c) & (grp == s)
        w[m] = len(y) / (len(cells) * m.sum())
    return w


def group_auc(y: np.ndarray, p: np.ndarray, grp: np.ndarray) -> float:
    """Mean over skin groups of the tumour vs non-tumour AUC (plain AUC when there is one group)."""
    return float(np.mean([roc_auc_score(y[grp == s], p[grp == s]) for s in np.unique(grp)]))


def gate_sets(E, manifest, lists, tumours, label_col, classes):
    tum = manifest[manifest[label_col].isin(classes)]
    gl = pd.concat([pd.read_csv(f) for f in lists], ignore_index=True)
    sets = {}
    for name, tsplits in tumours.items():
        t = tum[tum["split"].isin(tsplits)]
        n = gl[gl["gate_split"] == name]
        X = np.vstack([E.loc[t["image_id"]].to_numpy(), E.loc[n["image_id"]].to_numpy()])
        y = np.r_[np.ones(len(t)), np.zeros(len(n))]
        grp = np.r_[skin(t["fst_bucket"]), skin(n["fst_bucket"])]
        sets[name] = (X, y, grp)
    return sets, gl


def fit_gate(sets, Cs, weightings):
    Xtr, ytr, gtr = sets["train"]
    Xva, yva, gva = sets["val"]
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    fits, val = {}, {}
    for wt in weightings:
        for c in Cs:
            if wt == "class":
                f = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(Ztr, ytr)
            else:
                f = LogisticRegression(C=c, max_iter=5000).fit(Ztr, ytr, sample_weight=cell_weights(ytr, gtr))
            fits[wt, c] = f
            val[wt, c] = group_auc(yva, f.predict_proba(Zva)[:, 1], gva)
    best = max(val, key=val.get)
    print("gate val AUC (mean over skin groups):", {f"{w}/{c}": round(v, 4) for (w, c), v in val.items()},
          "->", best)
    clf = fits[best]
    gv = clf.predict_proba(Zva)[:, 1]
    q = {s: float(np.quantile(gv[(yva == 1) & (gva == s)], 1 - KEEP)) for s in np.unique(gva[yva == 1])}
    t95 = min(q.values())
    by_group = {str(s): round(roc_auc_score(yva[gva == s], gv[gva == s]), 4) for s in np.unique(gva)}
    print("gate val AUC by skin group:", by_group, "t95 by group:", {str(s): round(v, 4) for s, v in q.items()})
    return (lambda X: clf.predict_proba(scaler.transform(X))[:, 1]), best, t95, val[best]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--variant", choices=list(VARIANTS), default="e10")
    ap.add_argument("--splits", nargs="+", default=["source_val", "source_test", "target_val", "target_test", "ak_side"])
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    V = VARIANTS[args.variant]
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["manifest"], low_memory=False)
    label_col, classes = cfg["data"]["label_col"], runs.classes_of(cfg)
    E = load_embeddings(cfg["features_dir"])
    for f in V["lists"]:
        missing = (~pd.read_csv(f)["image_id"].isin(E.index)).sum()
        assert missing == 0, f"{missing} gate images without embeddings: run derm_foundation_embed.py --ids {f}"
    sets, gl = gate_sets(E, manifest, V["lists"], V["tumours"], label_col, classes)
    gate, (wt, C), t95, auc = fit_gate(sets, V["C"], V["weighting"])
    print(f"t95 = {t95:.4f} (keeps {KEEP:.0%} of val tumours)")

    cols = [f"p_{c}" for c in classes]
    malignant = [c for c in classes if c != "benign"]
    data, diag = {}, []
    for s in args.splits + args.final:
        df = manifest[manifest["split"] == s][["image_id", label_col, "label_binary", "fst_bucket"]].reset_index(drop=True)
        base = pd.read_csv(Path(cfg["runs_dir"]) / BASE / "s0" / "preds" / f"{s}.csv").set_index("image_id")
        g = gate(E.loc[df["image_id"]].to_numpy())
        P = gated(base.loc[df["image_id"], cols].to_numpy(), g, classes)
        data[s] = (df[["image_id"]], P, None)
        keep = g >= t95
        for fst in ["all"] + sorted(df["fst_bucket"].dropna().unique()):
            f = (df.index >= 0) if fst == "all" else (df["fst_bucket"] == fst).to_numpy()
            for grp, m in [("cancer", df[label_col].isin(malignant) | (df["label_binary"] == 1)),
                           ("benign tumour", df[label_col] == "benign"),
                           ("other", ~df[label_col].isin(classes) & (df["label_binary"] != 1)),
                           ("all", df.index >= 0)]:
                m = np.asarray(m) & f
                if m.sum():
                    diag.append({"split": s, "fst_bucket": fst, "group": grp, "n": int(m.sum()),
                                 "kept_by_gate": round(float(keep[m].mean()), 3),
                                 "median_g": round(float(np.median(g[m])), 3)})
    for name in ["test", "dark_test"]:
        ids = gl.loc[gl["gate_split"] == name, "image_id"]
        g = gate(E.loc[ids].to_numpy())
        diag.append({"split": f"fitz_nontumour_{name}", "fst_bucket": "all", "group": "other", "n": len(ids),
                     "kept_by_gate": round(float((g >= t95).mean()), 3), "median_g": round(float(np.median(g)), 3)})
    write_run(V["run"], 0, cfg, lambda X: X, data, args.splits, args.final, args.force,
              {"gate": "derm_foundation_lr", "gate_C": C, "gate_weighting": wt, "gate_val_auc": round(auc, 4),
               "gate_t95": round(t95, 4), "base": BASE})
    D = pd.DataFrame(diag)
    out = Path(cfg.get("results_dir", "results")) / V["run"]
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "gate_diagnostics.csv", index=False)
    print(D[D["fst_bucket"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
