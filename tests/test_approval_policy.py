"""Checks on the approval-policy view of the ranking results."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.approval_policy import (
    approval_policy_deltas,
    approval_policy_table,
    bootstrap_policy_delta,
    capture_curve,
    cutoff_at_approval_rate,
    fixed_cutoff_comparison,
    swap_analysis,
)


def _population(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-(latent - 2.0)))
    y = rng.binomial(1, probability)
    good = 1 / (1 + np.exp(-(1.6 * latent - 2.0)))
    # Noisier score on the same scale: weaker ranking, so decisions differ.
    noise = rng.normal(size=n)
    weak = 1 / (1 + np.exp(-(0.7 * latent + 1.2 * noise - 2.0)))
    return y, good, weak


def test_cutoff_hits_requested_fraction():
    scores = np.linspace(0, 1, 1001)

    cutoff = cutoff_at_approval_rate(scores, 0.30)

    assert (scores <= cutoff).mean() == pytest.approx(0.30, abs=0.01)


def test_cutoff_rejects_bad_rate():
    with pytest.raises(ValueError):
        cutoff_at_approval_rate(np.linspace(0, 1, 10), 1.0)


def test_policy_table_columns():
    y, good, weak = _population()

    table = approval_policy_table(
        y, {"cashflow": weak, "cashflow_plus_finrep": good}, [0.5, 0.8]
    )

    assert len(table) == 4
    assert table["approved_n"].between(0, len(y)).all()
    assert table["approved_default_rate"].between(0, 1).all()
    # Same approval level, so the better model should approve a safer book.
    bad = table.pivot(
        index="approval_rate", columns="model", values="approved_default_rate"
    )
    assert (bad["cashflow_plus_finrep"] < bad["cashflow"]).all()


def test_deltas_positive_when_challenger_better():
    y, good, weak = _population()

    table = approval_policy_table(
        y, {"cashflow": weak, "cashflow_plus_finrep": good}, [0.6]
    )
    row = approval_policy_deltas(table).iloc[0]

    assert row["defaults_avoided_per_10k_approved"] > 0
    assert row["delta_bad_rate_pp"] < 0
    assert row["capture_delta_pp"] > 0


def test_deltas_negative_when_challenger_worse():
    y, good, weak = _population()

    table = approval_policy_table(
        y, {"cashflow": good, "cashflow_plus_finrep": weak}, [0.6]
    )
    deltas = approval_policy_deltas(table)

    assert deltas.iloc[0]["defaults_avoided_per_10k_approved"] < 0


def test_bootstrap_brackets_estimate():
    y, good, weak = _population(n=3000)

    intervals = bootstrap_policy_delta(y, weak, good, [0.7], n_boot=120, seed=3)
    row = intervals.iloc[0]

    assert row["ci_2_5"] <= row["defaults_avoided_per_10k_approved"] <= row["ci_97_5"]
    assert 0.0 <= row["p_nonpositive"] <= 1.0
    assert row["n_boot_valid"] == 120


def test_swap_groups_are_balanced():
    y, good, weak = _population()

    swaps = swap_analysis(y, weak, good, 0.7)
    counts = swaps.set_index("group")["applicants"]

    # Anyone let in by the challenger displaces someone at the same rate.
    assert counts["swapped_out"] == counts["swapped_in"]
    assert counts.sum() > 0
    assert (counts.sum() / len(y)) < 0.5


def test_fixed_cutoff_reports_rates():
    y, good, weak = _population()

    cutoff = float(np.quantile(weak, 0.7))
    table = fixed_cutoff_comparison(
        y, {"cashflow": weak, "cashflow_plus_finrep": good}, cutoff
    )

    assert set(table["model"]) == {"cashflow", "cashflow_plus_finrep"}
    assert table["approval_rate"].between(0, 1).all()


def test_capture_curve_endpoints():
    y, good, _ = _population()

    curve = capture_curve(y, good)

    assert curve["share_rejected"].iloc[0] <= curve["share_rejected"].iloc[-1]
    assert curve["share_defaults_captured"].iloc[-1] == pytest.approx(1.0)
    assert curve["share_defaults_captured"].is_monotonic_increasing
    assert curve["share_defaults_captured"].iloc[0] > 0
