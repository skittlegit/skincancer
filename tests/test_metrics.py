import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from skincancer import metrics as M
from skincancer.labels import indian_prior

CLASSES = ["bcc", "scc", "mel", "benign"]


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4)) + 1.5 * np.eye(4)[y]
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y_bin = (y != 3).astype(float)
    return y, y_bin, P


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
    assert M.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(M.auc(np.ones(5), np.arange(5)))


def test_macro_auc_matches_sklearn_and_skips_absent_class():
    y, _, P = _data()
    assert M.macro_auc(y, P) == pytest.approx(roc_auc_score(y, P, multi_class="ovr", average="macro"))
    m = y != 2  # drop all melanoma
    expected = np.mean([roc_auc_score(y[m] == k, P[m][:, k]) for k in (0, 1, 3)])
    assert M.macro_auc(y[m], P[m]) == pytest.approx(expected)


def test_confusion_rates():
    P = np.array([[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]])
    y_bin = np.array([1, 1, 0, 0.0])
    assert M.missed_cancer_rate(y_bin, P, CLASSES) == 0.5
    assert M.false_alarm_rate(y_bin, P, CLASSES) == 0.5


def test_bootstrap_ci_brackets_point_and_seeds_averaged():
    y, y_bin, P = _data()
    _, _, P2 = _data(seed=5)
    groups = np.arange(len(y)) // 2
    fns = M.metric_fns(CLASSES, "multiclass")
    out = M.bootstrap(fns, y, y_bin, [P, P2], groups, n_boot=200)
    r = out["macro_auc"]
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]
    assert r["mean"] == pytest.approx((M.macro_auc(y, P) + M.macro_auc(y, P2)) / 2)
    assert r["std"] > 0


def test_prior_shift_identity_and_direction():
    _, _, P = _data()
    u = np.full(4, 0.25)
    assert np.allclose(M.prior_shift(P, u, u), P)
    to = indian_prior(CLASSES, u, "consensus")
    Q = M.prior_shift(P, u, to)
    assert np.allclose(Q.sum(1), 1)
    assert Q[:, 0].mean() > P[:, 0].mean() and Q[:, 2].mean() < P[:, 2].mean()   # BCC up, MEL down
    assert Q[:, 3].mean() == pytest.approx(P[:, 3].mean(), abs=0.02)            # benign ~kept


def test_indian_prior_keeps_benign_share():
    p = indian_prior(CLASSES, [10, 10, 30, 50], "consensus")
    assert p[3] == pytest.approx(0.5)
    assert p[:3] / p[:3].sum() == pytest.approx([0.45, 0.40, 0.15])
    p3 = indian_prior(["nmsc", "mel", "benign"], [1, 1, 2])
    assert p3 == pytest.approx([0.5 * 0.85, 0.5 * 0.15, 0.5])


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
