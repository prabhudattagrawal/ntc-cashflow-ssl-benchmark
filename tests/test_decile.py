import numpy as np

from src.decile import decile_analysis, pentile_analysis


def _sample():
    probabilities = np.linspace(0.01, 0.99, 100)
    labels = (np.arange(100) % 4 == 0).astype(int)
    return labels, probabilities


def test_capture_is_monotone():
    labels, probabilities = _sample()

    deciles = decile_analysis(labels, probabilities, n_bins=10)
    cumulative = deciles["cumulative_default_capture_pct"].to_numpy()

    assert np.all(np.diff(cumulative) >= 0)
    assert cumulative[-1] == 100.0


def test_pentiles_are_equal_sized():
    labels, probabilities = _sample()

    pentiles = pentile_analysis(labels, probabilities)

    assert pentiles["pentile"].tolist() == [1, 2, 3, 4, 5]
    assert pentiles["applicants"].tolist() == [20, 20, 20, 20, 20]
    assert pentiles["cumulative_default_capture_pct"].iloc[-1] == 100.0
