"""E10: a "not a tumour" gate in front of the Derm Foundation classifier (D26).

The gate is a class-balanced logistic regression on frozen Derm Foundation embeddings, tumour vs
non-tumour. Tumours are source_train (FST I–III); non-tumours are Fitzpatrick17k images outside
the four classes (data/gate/nontumour.csv). C is chosen on gate val AUC. The system multiplies the
gate's P(tumour) into e1_fm_lr's cancer probabilities and is written as a normal run (e10_gate),
so evaluate.py scores it like any other experiment.

    python -m skincancer.gate
    python -m skincancer.gate --final ddi_test dermacon_test dermacon_other pad_dark_test midas_test
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
from .foundation import LR_C, load_embeddings, write_run

GATE_LIST = "data/gate/nontumour.csv"
BASE = "e1_fm_lr"
KEEP = 0.95          # the cut-off t95 keeps this share of source_val tumours


def gated(P: np.ndarray, g: np.ndarray, classes: list[str]) -> np.ndarray:
    """p'(cancer) = g·p(cancer); p'(benign) = 1 − g·Σ p(cancer). Rows still sum to 1."""
    b = classes.index("benign")
    out = P * g[:, None]
    out[:, b] = 1.0 - (out.sum(1) - out[:, b])
    return out


def fit_gate(E: pd.DataFrame, manifest: pd.DataFrame, gl: pd.DataFrame, label_col: str, classes: list[str]):
    tum = manifest[manifest[label_col].isin(classes)]
    sets = {}
    for name, tsplit in [("train", "source_train"), ("val", "source_val")]:
        t_ids = tum.loc[tum["split"] == tsplit, "image_id"]
        n_ids = gl.loc[gl["gate_split"] == name, "image_id"]
        X = np.vstack([E.loc[t_ids].to_numpy(), E.loc[n_ids].to_numpy()])
        y = np.r_[np.ones(len(t_ids)), np.zeros(len(n_ids))]
        sets[name] = (X, y, len(t_ids))
    Xtr, ytr, _ = sets["train"]
    Xva, yva, n_tum_val = sets["val"]
    scaler = StandardScaler().fit(Xtr)
    fits = {c: LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(scaler.transform(Xtr), ytr)
            for c in LR_C}
    val = {c: roc_auc_score(yva, f.predict_proba(scaler.transform(Xva))[:, 1]) for c, f in fits.items()}
    best = max(val, key=val.get)
    print("gate val AUC (tumour vs non-tumour):", {c: round(v, 4) for c, v in val.items()}, "-> C =", best)
    clf = fits[best]
    g_val_tum = clf.predict_proba(scaler.transform(Xva[:n_tum_val]))[:, 1]
    t95 = float(np.quantile(g_val_tum, 1 - KEEP))
    return (lambda X: clf.predict_proba(scaler.transform(X))[:, 1]), best, t95, val[best]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--splits", nargs="+", default=["source_val", "source_test", "target_val", "target_test", "ak_side"])
    ap.add_argument("--final", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg["manifest"], low_memory=False)
    label_col, classes = cfg["data"]["label_col"], runs.classes_of(cfg)
    E = load_embeddings(cfg["features_dir"])
    gl = pd.read_csv(GATE_LIST)
    missing = (~gl["image_id"].isin(E.index)).sum()
    assert missing == 0, f"{missing} gate images without embeddings: run derm_foundation_embed.py --ids {GATE_LIST}"
    gate, C, t95, auc = fit_gate(E, manifest, gl, label_col, classes)
    print(f"t95 = {t95:.4f} (keeps {KEEP:.0%} of source_val tumours)")

    cols = [f"p_{c}" for c in classes]
    malignant = [c for c in classes if c != "benign"]
    data, diag = {}, []
    for s in args.splits + args.final:
        df = manifest[manifest["split"] == s][["image_id", label_col, "label_binary"]].reset_index(drop=True)
        base = pd.read_csv(Path(cfg["runs_dir"]) / BASE / "s0" / "preds" / f"{s}.csv").set_index("image_id")
        g = gate(E.loc[df["image_id"]].to_numpy())
        P = gated(base.loc[df["image_id"], cols].to_numpy(), g, classes)
        data[s] = (df[["image_id"]], P, None)
        keep = g >= t95
        for grp, m in [("cancer", df[label_col].isin(malignant) | (df["label_binary"] == 1)),
                       ("benign tumour", df[label_col] == "benign"),
                       ("other", ~df[label_col].isin(classes) & (df["label_binary"] != 1)), ("all", df.index >= 0)]:
            if m.sum():
                diag.append({"split": s, "group": grp, "n": int(m.sum()), "kept_by_gate": round(float(keep[m].mean()), 3)})
    for name in ["test", "dark_test"]:
        ids = gl.loc[gl["gate_split"] == name, "image_id"]
        diag.append({"split": f"fitz_nontumour_{name}", "group": "other", "n": len(ids),
                     "kept_by_gate": round(float((gate(E.loc[ids].to_numpy()) >= t95).mean()), 3)})
    write_run("e10_gate", 0, cfg, lambda X: X, data, args.splits, args.final, args.force,
              {"gate": "derm_foundation_lr", "gate_C": C, "gate_val_auc": round(auc, 4), "gate_t95": round(t95, 4),
               "base": BASE})
    D = pd.DataFrame(diag)
    out = Path(cfg.get("results_dir", "results")) / "e10_gate"
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "gate_diagnostics.csv", index=False)
    print(D.to_string(index=False))


if __name__ == "__main__":
    main()
