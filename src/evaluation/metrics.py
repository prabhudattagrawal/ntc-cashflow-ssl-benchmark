import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ks_statistic(y_true, probability):
    y = np.asarray(y_true)
    p = np.asarray(probability)
    y = y[np.argsort(p)]

    bad = y.sum()
    good = len(y) - bad
    if bad == 0 or good == 0:
        return float("nan")

    cum_bad = np.cumsum(y) / bad
    cum_good = np.cumsum(1 - y) / good
    return float(np.max(np.abs(cum_bad - cum_good)))


def expected_calibration_error(y_true, probability, n_bins=10):
    y = np.asarray(y_true)
    p = np.asarray(probability)
    edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            in_bin = (p >= edges[i]) & (p < edges[i + 1])
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def evaluate(y_true, probability):
    if len(np.unique(y_true)) < 2:
        auc, pr = float("nan"), float("nan")
    else:
        auc = roc_auc_score(y_true, probability)
        pr = average_precision_score(y_true, probability)

    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr),
        "ks": ks_statistic(y_true, probability),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece_10": expected_calibration_error(y_true, probability, 10),
        "default_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def psi(expected, actual, buckets=10, eps=1e-6):
    """Population stability index for a score or feature, binned on expected."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(cuts) < 3:
        return float("nan")
    cuts[0], cuts[-1] = -np.inf, np.inf

    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))
