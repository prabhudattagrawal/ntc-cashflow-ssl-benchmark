import numpy as np
from scipy.stats import entropy, ks_2samp

from src.evaluation.metrics import evaluate, ks_statistic, psi


def test_metrics_stay_in_range():
    y = [0, 0, 1, 1]
    p = [.1, .2, .8, .9]

    assert 0 <= ks_statistic(y, p) <= 1

    result = evaluate(y, p)
    assert result["roc_auc"] == 1.0
    assert result["brier"] < .1

    # A distribution compared with itself has no stability problem.
    assert psi(p, p) < 1e-8


def test_ks_matches_scipy():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([.05, .72, .21, .91, .35, .63, .42, .84])

    expected = ks_2samp(
        probabilities[labels == 1], probabilities[labels == 0]
    ).statistic

    assert ks_statistic(labels, probabilities) == expected


def test_psi_matches_symmetric_kl():
    expected_scores = np.linspace(0.001, 0.999, 1000)
    actual_scores = np.random.default_rng(42).beta(2, 2, size=1000)
    buckets = 5

    cuts = np.unique(np.quantile(expected_scores, np.linspace(0, 1, buckets + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    expected = np.histogram(expected_scores, bins=cuts)[0] / len(expected_scores)
    actual = np.histogram(actual_scores, bins=cuts)[0] / len(actual_scores)

    # PSI is KL in both directions added together.
    scipy_psi = entropy(actual, expected) + entropy(expected, actual)

    assert np.isclose(psi(expected_scores, actual_scores, buckets=buckets), scipy_psi)
