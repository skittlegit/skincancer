"""Metrics, group bootstrap over seeds (D8) and prior-shift correction (D7).

Everything here works on arrays: y (class index, -1 = not in the class set), y_bin
(1 malignant / 0 benign / nan), P (n x K probabilities), and the class list.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .labels import MALIGNANT


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties averaged). NaN if one class is empty."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(y: np.ndarray, P: np.ndarray) -> float:
    """Macro one-vs-rest AUC over classes present in y (D8). Rows with y = -1 are dropped."""
    keep = y >= 0
    y, P = y[keep], P[keep]
    vals = [auc(y == k, P[:, k]) for k in range(P.shape[1]) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(vals)) if len(vals) >= 2 else np.nan


def p_malignant(P: np.ndarray, classes: list[str]) -> np.ndarray:
    cols = [i for i, c in enumerate(classes) if c in MALIGNANT]
    return P[:, cols].sum(1)


def pred_malignant(P: np.ndarray, classes: list[str]) -> np.ndarray:
    return P.argmax(1) != classes.index("benign")


def missed_cancer_rate(y_bin, P, classes) -> float:
    """% of malignant images predicted benign."""
    m = y_bin == 1
    return float((~pred_malignant(P[m], classes)).mean()) if m.any() else np.nan


def false_alarm_rate(y_bin, P, classes) -> float:
    """% of benign lookalikes predicted malignant."""
    b = y_bin == 0
    return float(pred_malignant(P[b], classes).mean()) if b.any() else np.nan


def recall(y, P, k) -> float:
    m = y == k
    return float((P[m].argmax(1) == k).mean()) if m.any() else np.nan


def precision(y, P, k) -> float:
    m = (P.argmax(1) == k) & (y >= 0)
    return float((y[m] == k).mean()) if m.any() else np.nan


def balanced_accuracy(y, P) -> float:
    keep = y >= 0
    present = np.unique(y[keep])
    return float(np.mean([recall(y, P, k) for k in present])) if len(present) else np.nan


def metric_fns(classes: list[str], task: str) -> dict:
    """name -> f(y, y_bin, P). task: 'multiclass' (4/3-class labels) or 'binary'."""
    fns = {
        "binary_auc": lambda y, yb, P: auc(yb[~np.isnan(yb)] == 1, p_malignant(P, classes)[~np.isnan(yb)]),
        "missed_cancer_rate": lambda y, yb, P: missed_cancer_rate(yb, P, classes),
        "false_alarm_rate": lambda y, yb, P: false_alarm_rate(yb, P, classes),
    }
    if task == "multiclass":
        fns["macro_auc"] = lambda y, yb, P: macro_auc(y, P)
        fns["balanced_acc"] = lambda y, yb, P: balanced_accuracy(y, P)
        for k, c in enumerate(classes):
            fns[f"recall_{c}"] = lambda y, yb, P, k=k: recall(y, P, k)
            fns[f"precision_{c}"] = lambda y, yb, P, k=k: precision(y, P, k)
    return fns


def bootstrap(fns: dict, y, y_bin, Ps: list[np.ndarray], groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Seed-mean point estimate, std across seeds, and group-bootstrap percentile 95% CI of
    the seed-mean (same resample indices for every seed)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = {}
    per_seed = {n: np.array([f(y, y_bin, P) for P in Ps]) for n, f in fns.items()}
    boots = {n: np.full(n_boot, np.nan) for n in fns}
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for n, f in fns.items():
            vals = [f(y[idx], y_bin[idx], P[idx]) for P in Ps]
            boots[n][b] = np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan
    for n in fns:
        v = per_seed[n]
        bs = boots[n][~np.isnan(boots[n])]
        out[n] = {
            "mean": float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan,
            "std": float(np.nanstd(v)) if len(v) > 1 and not np.all(np.isnan(v)) else np.nan,
            "ci_lo": float(np.percentile(bs, 2.5)) if len(bs) else np.nan,
            "ci_hi": float(np.percentile(bs, 97.5)) if len(bs) else np.nan,
            "n_boot_valid": int(len(bs)),
        }
    return out


def prior_shift(P: np.ndarray, pi_from, pi_to) -> np.ndarray:
    """p_new(y|x) ∝ p(y|x) * pi_to(y) / pi_from(y), renormalised (Saerens et al. 2002)."""
    w = np.asarray(pi_to, float) / np.asarray(pi_from, float)
    Q = P * w[None, :]
    return Q / Q.sum(1, keepdims=True)


def confusion(y, P, k: int) -> np.ndarray:
    keep = y >= 0
    pred = P[keep].argmax(1)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y[keep], pred), 1)
    return cm
