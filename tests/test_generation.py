"""Sanity checks on the generator: dates, balances, stress and prevalence."""
import numpy as np
import pandas as pd

from src.data.generate_data import GENERATOR_ONLY_COLUMNS, generate_data


def test_generator_respects_dates_and_balances():
    applicants, transactions = generate_data(
        n_applicants=500, months_history=4, seed=13
    )

    dates = applicants.set_index("applicant_id")["application_date"]
    # Nothing may land on or after the application date.
    assert (transactions["transaction_date"] < transactions["applicant_id"].map(dates)).all()

    assert transactions["account_type"].notna().all()
    assert (
        transactions["transaction_date"] >= transactions["account_open_date"]
    ).all()
    assert applicants["account_tenure_months"].between(1, 120).all()
    assert applicants["account_type"].isin(["CHECKING", "SAVINGS", "WALLET"]).all()
    assert applicants["housing_status"].isin(
        ["RENTING", "MORTGAGE", "OWNED_OUTRIGHT", "FAMILY"]
    ).all()
    assert 0.02 < applicants["default_12m"].mean() < 0.25

    # Walk each account so a broken running balance shows up.
    for _, rows in transactions.groupby("applicant_id", sort=False):
        assert rows["transaction_date"].is_monotonic_increasing
        opening = rows["balance_after"].iloc[0] - rows["amount"].iloc[0]
        np.testing.assert_allclose(
            rows["balance_after"], opening + rows["amount"].cumsum().to_numpy()
        )


def test_variable_history_with_maturity_gap():
    applicants, transactions = generate_data(
        n_applicants=80,
        months_history=8,
        min_history_months=4,
        variable_history=True,
        maturity_gap_months=2,
        seed=5,
    )

    months = (
        transactions.assign(month=transactions["transaction_date"].dt.to_period("M"))
        .groupby("applicant_id")["month"]
        .nunique()
    )
    assert months.min() >= 4
    assert months.max() <= 8
    # Variable history is the point, so the counts should not all be the same.
    assert months.nunique() > 1

    cutoff = applicants.set_index("applicant_id")["application_date"].map(
        lambda d: d.to_period("M").to_timestamp() - pd.DateOffset(months=2)
    )
    latest = transactions.groupby("applicant_id")["transaction_date"].max()
    assert (latest < cutoff.loc[latest.index]).all()


def test_drift_only_affects_later_applicants():
    common = dict(
        n_applicants=200,
        months_history=3,
        start_application_date="2024-01-01",
        end_application_date="2026-12-01",
        seed=19,
    )
    baseline, _ = generate_data(**common)
    drifted, _ = generate_data(
        **common,
        drift_start_application_date="2025-07-01",
        drift_income_multiplier=2.0,
        drift_obligation_delta=0.1,
    )

    both = baseline[["applicant_id", "application_date", "base_income"]].merge(
        drifted[["applicant_id", "base_income"]],
        on="applicant_id",
        suffixes=("_base", "_drift"),
    )
    early = both["application_date"] < pd.Timestamp("2025-07-01")

    np.testing.assert_array_equal(
        both.loc[early, "base_income_base"], both.loc[early, "base_income_drift"]
    )
    np.testing.assert_allclose(
        both.loc[~early, "base_income_drift"],
        2 * both.loc[~early, "base_income_base"],
    )


def test_audit_latents_stay_out_of_reach():
    applicants, _ = generate_data(n_applicants=200, months_history=6, seed=21)

    for column in GENERATOR_ONLY_COLUMNS:
        assert column in applicants.columns

    assert applicants["p_default_conditional"].between(0, 1).all()
    assert applicants["p_default_realized"].between(0, 1).all()
    assert np.allclose(
        applicants["default_probability"], applicants["p_default_realized"]
    )
    # The realised probability carries the noise draw, so it cannot equal the
    # conditional one - otherwise the oracle would be the same thing.
    assert not np.allclose(
        applicants["p_default_conditional"], applicants["p_default_realized"]
    )
    assert applicants["default_12m"].isin([0, 1]).all()


def test_stress_shows_up_in_timing_only():
    applicants, transactions = generate_data(
        n_applicants=800, months_history=8, seed=31
    )
    stress = applicants.set_index("applicant_id")["sequence_stress_latent"]

    tx = transactions.copy()
    tx["day"] = tx["transaction_date"].dt.day
    tx["quintile"] = (
        pd.qcut(stress, 5, labels=False).reindex(tx["applicant_id"]).to_numpy()
    )

    rent = tx.loc[tx["category"].eq("RENT")].groupby("quintile")["day"].mean()
    salary = tx.loc[tx["category"].eq("SALARY")].groupby("quintile")["day"].mean()

    assert rent.iloc[-1] > rent.iloc[0] + 3
    assert salary.iloc[-1] > salary.iloc[0] + 2

    # ...but the monthly totals the cash-flow baseline sees should not move.
    monthly_rent = (
        tx.loc[tx["category"].eq("RENT")]
        .assign(month=tx["transaction_date"].dt.to_period("M"))
        .groupby(["applicant_id", "month"])["amount"].sum()
        .groupby("applicant_id").mean()
    )
    assert abs(stress.loc[monthly_rent.index].corr(monthly_rent.abs())) < 0.25


def test_intercept_moves_the_default_rate():
    kwargs = dict(n_applicants=1200, months_history=3, seed=21)
    high, _ = generate_data(base_default_logit=-1.5, **kwargs)
    low, _ = generate_data(base_default_logit=-3.5, **kwargs)

    high_rate = high["default_12m"].mean()
    low_rate = low["default_12m"].mean()

    assert low_rate < high_rate
    assert 0.0 <= low_rate < 0.15
    assert high_rate > 0.20
