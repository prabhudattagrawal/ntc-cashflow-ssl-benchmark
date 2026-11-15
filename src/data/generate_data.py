"""Synthetic applicants, pre-application transactions and 12-month default labels.

Two things are simulated on purpose. Cash-flow drivers (obligation, income
volatility, opening liquidity) show up in monthly totals, so the conventional
feature baseline can see them. Sequence-stress drivers (payment lateness, income
timing, spending tempo) only change when transactions happen, so they are
visible in the transaction sequence but not in the monthly aggregates.

Everything used only for auditing is listed in GENERATOR_ONLY_COLUMNS and gets
dropped before modelling.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "SALARY", "BUSINESS_INCOME", "RENT", "GROCERIES", "UTILITIES",
    "TRAVEL", "ONLINE_SHOPPING", "MEDICAL", "TRANSFER", "OTHER",
]
PROFILES = [
    "SALARIED_STABLE", "SALARIED_VOLATILE", "SELF_EMPLOYED",
    "HIGH_OBLIGATION", "LOW_LIQUIDITY",
]
ACCOUNT_TYPES = ["CHECKING", "SAVINGS", "WALLET"]
HOUSING_STATUSES = ["RENTING", "MORTGAGE", "OWNED_OUTRIGHT", "FAMILY"]

# Simulation bookkeeping, never model inputs. Kept in one place so the notebook
# and the CLI exclude the same columns.
GENERATOR_ONLY_COLUMNS = [
    "account_open_date",
    "profile",
    "base_income",
    "account_type",
    "housing_status",
    "housing_cost",
    "default_probability",
    "p_default_conditional",
    "p_default_realized",
    "obligation_latent",
    "income_cv_latent",
    "initial_balance_ratio_latent",
    "sequence_stress_latent",
    "payment_risk_latent",
    "income_timing_risk_latent",
    "spending_tempo_risk_latent",
    "transfer_reliance_risk_latent",
]

DAY_OFFSETS = np.arange(32, dtype="timedelta64[D]")

EXPENSE_CATEGORIES = ["GROCERIES", "UTILITIES", "TRAVEL",
                      "ONLINE_SHOPPING", "MEDICAL", "OTHER"]

FIXED_COST_MULTIPLIER = {
    "RENTING": 1.00,
    "MORTGAGE": 1.00,
    "OWNED_OUTRIGHT": 0.88,
    "FAMILY": 0.92,
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def generate_data(n_applicants=36_000, months_history=24,
                  start_application_date="2024-01-01",
                  end_application_date="2026-12-31", seed=42,
                  variable_history=False, min_history_months=None,
                  maturity_gap_months=0, drift_start_application_date=None,
                  drift_income_multiplier=1.0, drift_obligation_delta=0.0,
                  base_default_logit=-2.47):
    if months_history < 1:
        raise ValueError("months_history must be at least one.")
    if maturity_gap_months < 0:
        raise ValueError("maturity_gap_months must be non-negative.")
    if drift_income_multiplier <= 0:
        raise ValueError("drift_income_multiplier must be positive.")

    minimum_history = (min_history_months if min_history_months is not None
                       else max(1, months_history // 2))
    if variable_history and not 1 <= minimum_history <= months_history:
        raise ValueError("min_history_months must be between one and months_history.")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_application_date)
    end = pd.Timestamp(end_application_date)
    drift_start = (pd.Timestamp(drift_start_application_date)
                   if drift_start_application_date is not None else None)

    # Application volumes: a mean-reverting random walk, then largest-remainder
    # rounding so the monthly counts add up to n_applicants exactly.
    month_starts = pd.date_range(start, end, freq="MS")
    month_count = len(month_starts)
    log_probs = np.zeros(month_count)
    for m in range(1, month_count):
        log_probs[m] = 0.95 * log_probs[m - 1] + rng.uniform(-0.01, 0.01)
    probs = np.exp(log_probs - log_probs.max())
    probs = probs / probs.sum()

    monthly_counts = np.floor(probs * n_applicants).astype(int)
    remaining = n_applicants - monthly_counts.sum()
    if remaining > 0:
        fractional = probs * n_applicants - monthly_counts
        monthly_counts[np.argsort(fractional)[-remaining:]] += 1

    app_dates = []
    for m, count in enumerate(monthly_counts):
        app_dates.extend([month_starts[m]] * count)
    rng.shuffle(app_dates)

    # Everything below works on month indices and looks dates up in one table,
    # which is a lot cheaper than building a Timestamp per transaction.
    app_dates_arr = np.asarray(app_dates, dtype="datetime64[ns]")
    app_month_index = app_dates_arr.astype("datetime64[M]").astype("int64")
    max_back = max(121, months_history + maturity_gap_months) + 2
    span_lo = int(app_month_index.min()) - max_back
    span_hi = int(app_month_index.max()) + 2
    month_span = np.arange(span_lo, span_hi + 1, dtype="int64")
    month_first_by_index = month_span.astype("datetime64[M]").astype("datetime64[ns]")
    offset = span_lo

    base_expense_weights = np.array([.22, .13, .10, .09, .07, .05])
    base_expense_weights = base_expense_weights / base_expense_weights.sum()
    # Dirichlet concentration: how far an applicant can drift from the base mix.
    category_concentration = 10.0

    applicants = []
    transaction_rows = []
    initial_balances = np.empty(n_applicants, dtype="float64")

    for aid in range(n_applicants):
        profile = rng.choice(PROFILES, p=[.34, .16, .22, .16, .12])
        base_income = float(rng.lognormal(np.log(42000), .48))

        if profile == "SALARIED_STABLE":
            income_cv, obligation = .06, rng.uniform(.18, .38)
        elif profile == "SALARIED_VOLATILE":
            income_cv, obligation = .30, rng.uniform(.20, .48)
        elif profile == "SELF_EMPLOYED":
            income_cv, obligation = .38, rng.uniform(.18, .45)
        elif profile == "HIGH_OBLIGATION":
            income_cv, obligation = .12, rng.uniform(.48, .72)
        else:
            income_cv, obligation = .15, rng.uniform(.30, .52)

        # Stress risks are drawn independently of the cash-flow drivers. Beta(2,3)
        # has mean 0.4, so the index is centred by subtracting it.
        payment_risk = float(rng.beta(2.0, 3.0))
        income_timing_risk = float(rng.beta(2.0, 3.0))
        spending_tempo_risk = float(rng.beta(2.0, 3.0))
        transfer_reliance_risk = float(rng.beta(2.0, 3.0))
        sequence_stress = (
            0.45 * (payment_risk - 0.40)
            + 0.35 * (income_timing_risk - 0.40)
            + 0.20 * (spending_tempo_risk - 0.40)
        )

        app_date = app_dates[aid]
        app_month_pos = int(app_month_index[aid])
        account_tenure_months = int(
            rng.integers(max(1, minimum_history if variable_history else 1), 121)
        )
        account_open_date = pd.Timestamp(
            month_first_by_index[app_month_pos - account_tenure_months - offset]
        )
        account_type = str(rng.choice(ACCOUNT_TYPES, p=[.68, .22, .10]))
        housing_status = str(rng.choice(HOUSING_STATUSES, p=[.42, .18, .18, .22]))

        if drift_start is not None and app_date >= drift_start:
            base_income *= drift_income_multiplier
            obligation = float(np.clip(obligation + drift_obligation_delta, 0.01, 0.95))

        if housing_status == "RENTING":
            housing_cost = float(base_income * obligation * rng.uniform(.25, .55))
        elif housing_status == "MORTGAGE":
            housing_cost = float(base_income * obligation * rng.uniform(.25, .60))
        else:
            housing_cost = 0.0

        available_history_months = min(months_history, account_tenure_months)
        applicant_history_months = (
            int(rng.integers(minimum_history, available_history_months + 1))
            if variable_history else available_history_months
        )

        if profile == "LOW_LIQUIDITY":
            initial_balance = base_income * rng.uniform(.03, .22)
        elif profile == "HIGH_OBLIGATION":
            initial_balance = base_income * rng.uniform(.05, .40)
        else:
            initial_balance = base_income * rng.uniform(.10, .90)
        initial_balances[aid] = initial_balance
        initial_balance_ratio = initial_balance / (base_income + 1)

        expense_weights = rng.dirichlet(base_expense_weights * category_concentration)
        fixed_costs = obligation * FIXED_COST_MULTIPLIER[housing_status]
        self_employed = profile == "SELF_EMPLOYED"
        month_base = app_month_pos - offset

        for m in range(applicant_history_months):
            # The last generated month is the month right before application.
            month_idx = month_base - (
                applicant_history_months - m + maturity_gap_months
            )
            trend = 1.0 - 0.006 * m
            income = base_income * trend * max(.25, rng.normal(1.0, income_cv))

            # Stress is a bit more visible in the last quarter before applying.
            recency = 1.35 if m >= applicant_history_months - 3 else 1.0
            payment_late_days = int(round(payment_risk * 15 * recency))
            income_day_shift = int(round(income_timing_risk * 16 * recency))
            expense_upper = max(8, int(round(28 - 17 * spending_tempo_risk * recency)))

            if self_employed:
                n_income = int(rng.integers(2, 5))
                for _ in range(n_income):
                    amount = max(500.0, income / n_income * rng.uniform(.65, 1.35))
                    day = min(27, 1 + int(rng.integers(0, 26))
                              + int(round(income_day_shift * .6)))
                    transaction_rows.append(
                        (aid, month_idx, day, amount, "BUSINESS_INCOME", 1, None)
                    )
            else:
                day = min(27, 1 + int(rng.integers(0, 5)) + income_day_shift)
                transaction_rows.append(
                    (aid, month_idx, day, income, "SALARY", 1, None)
                )

            # Housing is a real transaction. Stressed applicants pay it later in
            # the month but the amount is unchanged, so monthly features miss it.
            expense_total = income * fixed_costs
            if housing_cost > 0:
                housing_payment = min(
                    housing_cost * rng.lognormal(0, .06), expense_total * .75
                )
                day = min(28, 1 + int(rng.integers(0, 5)) + payment_late_days)
                transaction_rows.append(
                    (aid, month_idx, day, -housing_payment, "RENT", 0, housing_status)
                )
            else:
                housing_payment = 0.0

            # The rest of the obligation is split with the applicant's own
            # weights. Spending tempo decides whether it lands early in the month.
            non_housing_total = max(0.0, expense_total - housing_payment)
            for category, weight in zip(EXPENSE_CATEGORIES, expense_weights):
                amount = max(50.0, non_housing_total * weight * rng.lognormal(0, .18))
                if category == "UTILITIES":
                    day = min(28, 1 + int(rng.integers(0, 5)) + payment_late_days)
                else:
                    day = int(rng.integers(2, expense_upper))
                transaction_rows.append(
                    (aid, month_idx, day, -amount, category, 0, None)
                )

            # Transfers add noise and volume. They are not part of the label.
            if rng.random() < 0.50 + 0.25 * transfer_reliance_risk:
                amount = float(
                    rng.uniform(500, 9000) * (1.0 + 0.5 * transfer_reliance_risk)
                )
                is_credit = bool(rng.random() < (0.30 + 0.25 * transfer_reliance_risk))
                day = int(rng.integers(2, 28))
                transaction_rows.append(
                    (aid, month_idx, day,
                     amount if is_credit else -amount, "TRANSFER", int(is_credit), None)
                )

        # Outcome = behavioural drivers + sequence stress + noise. The conditional
        # probability drops the noise draw and is the simulator-only oracle.
        latent_signal = (
            1.45 * (obligation - .32)
            + 1.25 * (income_cv - .12)
            + .55 * (initial_balance_ratio < .25)
            + 1.80 * sequence_stress
        )
        p_default_conditional = float(sigmoid(base_default_logit + 2.5 * latent_signal))
        noise = float(rng.normal(0, .42))
        p_default_realized = float(
            sigmoid(base_default_logit + 2.5 * (latent_signal + noise))
        )
        default = int(rng.binomial(1, p_default_realized))

        applicants.append({
            "applicant_id": aid,
            "application_date": app_date,
            "account_open_date": account_open_date,
            "account_tenure_months": account_tenure_months,
            "account_type": account_type,
            "housing_status": housing_status,
            "housing_cost": housing_cost,
            "profile": profile,
            "base_income": base_income,
            "default_12m": default,
            # Old name, kept because older notebooks still read it.
            "default_probability": p_default_realized,
            "p_default_conditional": p_default_conditional,
            "p_default_realized": p_default_realized,
            "obligation_latent": float(obligation),
            "income_cv_latent": float(income_cv),
            "initial_balance_ratio_latent": float(initial_balance_ratio),
            "sequence_stress_latent": float(sequence_stress),
            "payment_risk_latent": float(payment_risk),
            "income_timing_risk_latent": float(income_timing_risk),
            "spending_tempo_risk_latent": float(spending_tempo_risk),
            "transfer_reliance_risk_latent": float(transfer_reliance_risk),
        })

    raw = pd.DataFrame(
        transaction_rows,
        columns=["applicant_id", "month_idx", "day", "amount",
                 "category", "is_credit", "housing_payment_type"],
    )
    # The row list is large at scale, drop it before we start sorting.
    del transaction_rows
    raw = raw.sort_values(
        ["applicant_id", "month_idx", "day"], kind="stable"
    ).reset_index(drop=True)

    amounts = raw["amount"].to_numpy()
    applicant_ids = raw["applicant_id"].to_numpy()

    # Running balance per applicant, all vectorised: cumulative sum minus the sum
    # before each applicant's block, plus that applicant's opening balance.
    cumsum = np.cumsum(amounts)
    new_group = np.empty(len(applicant_ids), dtype=bool)
    new_group[0] = True
    np.not_equal(applicant_ids[1:], applicant_ids[:-1], out=new_group[1:])
    group_starts = np.flatnonzero(new_group)
    prior = cumsum[group_starts] - amounts[group_starts]
    group_id = np.cumsum(new_group) - 1
    balance_after = cumsum - prior[group_id] + initial_balances[applicant_ids]

    transaction_dates = (
        month_first_by_index[raw["month_idx"].to_numpy()]
        + DAY_OFFSETS[raw["day"].to_numpy() - 1]
    )

    tx = pd.DataFrame({
        "applicant_id": applicant_ids,
        "transaction_date": transaction_dates,
        "amount": amounts,
        "category": raw["category"].to_numpy(),
        "is_credit": raw["is_credit"].to_numpy(),
        "balance_after": balance_after,
        "housing_payment_type": raw["housing_payment_type"].to_numpy(),
    })

    app = pd.DataFrame(applicants)
    # Reuse whatever resolution pandas inferred above so both frames agree.
    date_dtype = app["application_date"].dtype
    app["account_open_date"] = app["account_open_date"].astype(date_dtype)
    tx["transaction_date"] = tx["transaction_date"].astype(date_dtype)
    del raw

    # Column maps rather than a join, a join would duplicate the whole frame.
    metadata = app.set_index("applicant_id")
    tx["account_type"] = tx["applicant_id"].map(metadata["account_type"])
    tx["account_open_date"] = tx["applicant_id"].map(metadata["account_open_date"])

    # No transaction may land on or after its own application date.
    application_date = tx["applicant_id"].map(metadata["application_date"])
    assert (tx["transaction_date"] < application_date).all()

    return app, tx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-applicants", type=int, default=4000)
    parser.add_argument("--months-history", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/generated")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    app, tx = generate_data(args.n_applicants, args.months_history, seed=args.seed)
    app.to_parquet(out / "applicants.parquet", index=False)
    tx.to_parquet(out / "transactions.parquet", index=False)
    print(f"Wrote {len(app):,} applicants and {len(tx):,} transactions to {out}")


if __name__ == "__main__":
    main()
