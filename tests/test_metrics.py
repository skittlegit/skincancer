import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from skincancer import metrics as M

CLASSES = ["not_cancer", "cancer"]


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    logits = rng.normal(size=(n, 2)) + 1.5 * np.eye(2)[y]
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, y.astype(float), P


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert M.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(M.auc(np.ones(5), np.arange(5)))


def test_cancer_probability_and_rates():
    P = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
    assert np.allclose(M.p_malignant(P, CLASSES), [0.3, 0.8, 0.6, 0.1])
    y_bin = np.array([1, 1, 0, 0.0])
    assert M.missed_cancer_rate(y_bin, P, CLASSES) == 0.5
    assert M.false_alarm_rate(y_bin, P, CLASSES) == 0.5


def test_bootstrap_ci_brackets_point_and_seeds_averaged():
    y, y_bin, P = _data()
    _, _, P2 = _data(seed=5)
    groups = np.arange(len(y)) // 2
    out = M.bootstrap(M.metric_fns(CLASSES), y, y_bin, [P, P2], groups, n_boot=200)
    r = out["binary_auc"]
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]
    assert r["mean"] == pytest.approx((roc_auc_score(y, P[:, 1]) + roc_auc_score(y, P2[:, 1])) / 2)
    assert r["std"] > 0
    y_bin[:5] = np.nan                                  # images with no label are left out
    assert M.metric_fns(CLASSES)["binary_auc"](y, y_bin, P) == pytest.approx(roc_auc_score(y[5:], P[5:, 1]))


def test_sensitivity_threshold_and_operating_point():
    from skincancer.metrics import operating_point_fns, sensitivity_threshold
    yb = np.array([1] * 10 + [0] * 5, float)
    p = np.r_[np.linspace(0.95, 0.5, 10), [0.9, 0.6, 0.4, 0.2, 0.1]]
    t = sensitivity_threshold(yb, p, 0.9)
    assert np.isclose(t, p[8])                     # 9 of 10 cancers at or above t
    fns = operating_point_fns()
    Q = (p - t)[:, None]
    assert np.isclose(fns["missed_cancer_rate_op90"](None, yb, Q), 0.1)
    assert np.isclose(fns["false_alarm_rate_op90"](None, yb, Q), 0.4)   # t = 0.55: 0.9 and 0.6 pass
