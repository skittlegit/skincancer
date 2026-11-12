"""Metrics and the group bootstrap over seeds.

Everything here works on arrays: y (class index, -1 = no label), y_bin (1 cancer /
0 not cancer / nan), P (n x 2 probabilities), and the class list.
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


def p_malignant(P: np.ndarray, classes: list[str]) -> np.ndarray:
    cols = [i for i, c in enumerate(classes) if c in MALIGNANT]
    return P[:, cols].sum(1)


def pred_malignant(P: np.ndarray, classes: list[str]) -> np.ndarray:
    """True where the most probable class is a cancer class."""
    return np.isin(P.argmax(1), [i for i, c in enumerate(classes) if c in MALIGNANT])


def missed_cancer_rate(y_bin, P, classes) -> float:
    """% of malignant images predicted benign."""
    m = y_bin == 1
    return float((~pred_malignant(P[m], classes)).mean()) if m.any() else np.nan


def false_alarm_rate(y_bin, P, classes) -> float:
    """% of benign lookalikes predicted malignant."""
    b = y_bin == 0
    return float(pred_malignant(P[b], classes).mean()) if b.any() else np.nan


def sensitivity_threshold(y_bin, p_mal, sens: float) -> float:
    """Largest threshold t with share(p_mal >= t | malignant) >= sens."""
    s = np.sort(np.asarray(p_mal)[np.asarray(y_bin) == 1])[::-1]
    if not len(s):
        return np.nan
    return float(s[min(len(s) - 1, int(np.ceil(sens * len(s))) - 1)])


def operating_point_fns() -> dict:
    """Rates at a fixed threshold; P here is one column: p_malignant minus the seed's threshold."""
    def missed(y, yb, P):
        m = yb == 1
        return float((P[m, 0] < 0).mean()) if m.any() else np.nan

    def alarm(y, yb, P):
        b = yb == 0
        return float((P[b, 0] >= 0).mean()) if b.any() else np.nan

    return {"missed_cancer_rate_op90": missed, "false_alarm_rate_op90": alarm}


def metric_fns(classes: list[str]) -> dict:
    """name -> f(y, y_bin, P). Rates use the argmax decision (P(cancer) >= 0.5)."""
    return {
        "binary_auc": lambda y, yb, P: auc(yb[~np.isnan(yb)] == 1, p_malignant(P, classes)[~np.isnan(yb)]),
        "missed_cancer_rate": lambda y, yb, P: missed_cancer_rate(yb, P, classes),
        "false_alarm_rate": lambda y, yb, P: false_alarm_rate(yb, P, classes),
    }


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
