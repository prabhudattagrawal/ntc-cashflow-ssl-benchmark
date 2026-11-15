import pandas as pd

from src.features.cashflow_features import (
    build_cashflow_features,
    winsorize_extreme_features,
)


def test_future_transactions_excluded():
    applicants = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    tx = pd.DataFrame({
        "applicant_id": [1, 1],
        "transaction_date": [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-03-01")],
        "amount": [1000.0, 999999.0],
        "category": ["SALARY", "SALARY"],
        "is_credit": [1, 1],
        "balance_after": [1000.0, 1000999.0],
    })

    features = build_cashflow_features(applicants, tx)

    # The big one is dated on the application day, so it must not count.
    assert features.loc[0, "avg_monthly_credits"] == 1000.0


def test_no_target_column_in_features():
    applicants = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    tx = pd.DataFrame({
        "applicant_id": [1],
        "transaction_date": [pd.Timestamp("2025-02-01")],
        "amount": [1000.0],
        "category": ["SALARY"],
        "is_credit": [1],
        "balance_after": [1000.0],
    })

    features = build_cashflow_features(applicants, tx)

    assert "default_12m" not in features.columns
    assert "applicant_id" in features.columns


def test_opening_balance_uses_earliest_transaction():
    applicants = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    # Deliberately out of order: the February 1 row is the opening one.
    tx = pd.DataFrame({
        "applicant_id": [1, 1],
        "transaction_date": [pd.Timestamp("2025-02-02"), pd.Timestamp("2025-02-01")],
        "amount": [-50.0, 100.0],
        "category": ["GROCERIES", "SALARY"],
        "is_credit": [0, 1],
        "balance_after": [150.0, 200.0],
    })

    features = build_cashflow_features(applicants, tx)

    assert features.loc[0, "opening_balance"] == 100.0
    assert features.loc[0, "opening_balance_ratio"] == 100.0 / 101.0


def test_row_order_does_not_change_features():
    applicants = pd.DataFrame({
        "applicant_id": [1, 2],
        "application_date": [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01")],
    })
    tx = pd.DataFrame({
        "applicant_id": [1, 1, 2, 2],
        "transaction_date": [
            pd.Timestamp("2025-02-02"), pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-02-02"), pd.Timestamp("2025-02-01"),
        ],
        "amount": [-50.0, 100.0, -25.0, 80.0],
        "category": ["GROCERIES", "SALARY", "GROCERIES", "SALARY"],
        "is_credit": [0, 1, 0, 1],
        "balance_after": [150.0, 200.0, 135.0, 160.0],
    })

    expected = build_cashflow_features(applicants, tx)
    shuffled = build_cashflow_features(applicants, tx.sample(frac=1, random_state=4))

    pd.testing.assert_frame_equal(expected, shuffled)


def test_account_and_housing_features():
    applicants = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
        "housing_cost": [500.0],
    })
    tx = pd.DataFrame({
        "applicant_id": [1] * 6,
        "transaction_date": [
            pd.Timestamp("2024-12-02"), pd.Timestamp("2024-12-05"),
            pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05"),
            pd.Timestamp("2025-02-02"), pd.Timestamp("2025-02-05"),
        ],
        "amount": [2000.0, -500.0, 2000.0, -500.0, 2000.0, -500.0],
        "category": ["SALARY", "RENT", "SALARY", "RENT", "SALARY", "RENT"],
        "is_credit": [1, 0, 1, 0, 1, 0],
        "balance_after": [2000.0, 1500.0, 3500.0, 3000.0, 5000.0, 4500.0],
        "account_type": ["CHECKING"] * 6,
        "account_open_date": [pd.Timestamp("2024-03-01")] * 6,
        "housing_payment_type": [None, "RENTING", None, "RENTING", None, "RENTING"],
    })

    features = build_cashflow_features(applicants, tx).iloc[0]

    assert features["account_tenure_months"] == 12
    assert features["account_type_checking_share"] == 1.0
    assert features["has_housing_payment"] == 1
    assert features["avg_monthly_housing_payment"] == 500.0
    assert features["spend_housing_share"] == 0.25
    assert features["income_cv"] == 0.0
    assert features["expense_to_income_ratio"] == 500.0 / 2001.0
    assert features["income_event_frequency"] == 1.0
    assert features["transfer_in_ratio"] == 0.0
    assert features["transfer_out_ratio"] == 0.0

    # These were constant, duplicated or built from generator metadata, so they
    # should no longer be produced.
    removed_features = [
        "account_history_coverage",
        "account_type_count",
        "housing_payment_regularity",
        "housing_payment_events",
        "housing_payment_months",
        "housing_payment_total",
        "housing_payment_to_income_ratio",
        "balance_volatility_ratio",
        "income_gap_days",
        "net_monthly_cashflow",
    ]
    for name in removed_features:
        assert name not in features.index


def _two_month_applicant(applicant_id, extra_tx=None):
    tx = pd.DataFrame({
        "applicant_id": [applicant_id] * 4,
        "transaction_date": [
            pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-05"),
            pd.Timestamp("2025-02-02"), pd.Timestamp("2025-02-05"),
        ],
        "amount": [1000.0, -400.0, 1000.0, -400.0],
        "category": ["SALARY", "GROCERIES", "SALARY", "GROCERIES"],
        "is_credit": [1, 0, 1, 0],
        "balance_after": [1000.0, 600.0, 1600.0, 1200.0],
        "account_type": ["CHECKING"] * 4,
        "account_open_date": [pd.Timestamp("2024-01-01")] * 4,
    })
    if extra_tx is not None:
        tx = pd.concat([tx, pd.DataFrame(extra_tx)], ignore_index=True)
    return tx


def test_income_cv_only_uses_income():
    applicants = pd.DataFrame({
        "applicant_id": [1, 2],
        "application_date": [pd.Timestamp("2025-03-01")] * 2,
    })
    base = _two_month_applicant(1)
    with_transfer = _two_month_applicant(2, extra_tx={
        "applicant_id": [2],
        "transaction_date": [pd.Timestamp("2025-01-20")],
        "amount": [9000.0],
        "category": ["TRANSFER"],
        "is_credit": [1],
        "balance_after": [10600.0],
        "account_type": ["CHECKING"],
        "account_open_date": [pd.Timestamp("2024-01-01")],
    })
    features = build_cashflow_features(
        applicants, pd.concat([base, with_transfer], ignore_index=True)
    ).set_index("applicant_id")

    # Salary is flat across both months, so the transfer does not move the CV.
    assert features.loc[1, "income_cv"] == 0.0
    assert features.loc[2, "income_cv"] == 0.0
    assert features.loc[2, "avg_monthly_credits"] > features.loc[1, "avg_monthly_credits"]


def test_expense_ratio_ignores_transfers():
    applicants = pd.DataFrame({
        "applicant_id": [1, 2],
        "application_date": [pd.Timestamp("2025-03-01")] * 2,
    })
    base = _two_month_applicant(1)
    with_transfer = _two_month_applicant(2, extra_tx={
        "applicant_id": [2],
        "transaction_date": [pd.Timestamp("2025-01-20")],
        "amount": [-5000.0],
        "category": ["TRANSFER"],
        "is_credit": [0],
        "balance_after": [-4400.0],
        "account_type": ["CHECKING"],
        "account_open_date": [pd.Timestamp("2024-01-01")],
    })
    features = build_cashflow_features(
        applicants, pd.concat([base, with_transfer], ignore_index=True)
    ).set_index("applicant_id")

    assert features.loc[1, "expense_to_income_ratio"] == features.loc[
        2, "expense_to_income_ratio"
    ]
    assert features.loc[2, "transfer_out_ratio"] > 0
    assert features.loc[1, "transfer_out_ratio"] == 0


def test_winsorize_keeps_all_applicants():
    applicants = pd.DataFrame({
        "applicant_id": [1, 2, 3],
        "application_date": [pd.Timestamp("2025-03-01")] * 3,
    })
    tx = pd.concat(
        [_two_month_applicant(1), _two_month_applicant(2), _two_month_applicant(3)],
        ignore_index=True,
    )
    # Applicant 3 gets an enormous salary so the IQR bound has something to clip.
    tx.loc[(tx["applicant_id"] == 3) & (tx["category"] == "SALARY"), "amount"] = 1e7

    features = build_cashflow_features(applicants, tx)
    winsorized, bounds, clipped = winsorize_extreme_features(
        features, train_ids=[1, 2], columns=["avg_monthly_credits"]
    )

    assert len(winsorized) == len(features) == 3
    assert winsorized["applicant_id"].tolist() == [1, 2, 3]
    assert clipped["avg_monthly_credits"] == 1
    assert winsorized.loc[2, "avg_monthly_credits"] == bounds["avg_monthly_credits"]["upper"]
    assert features.loc[2, "avg_monthly_credits"] > winsorized.loc[2, "avg_monthly_credits"]
