"""Cash-flow features built only from transactions dated before application."""
import numpy as np
import pandas as pd

# One bucket per modelled expense category. No transfer bucket on purpose:
# moving money is not spending.
SPEND_CATEGORIES = {
    "housing": "RENT",
    "groceries": "GROCERIES",
    "utilities": "UTILITIES",
    "transport": "TRAVEL",
    "shopping": "ONLINE_SHOPPING",
    "medical": "MEDICAL",
    "other": "OTHER",
}
INCOME_CATEGORIES = ["SALARY", "BUSINESS_INCOME"]

# Columns the outlier helpers look at by default. The share features are the
# ones that blow up when income is tiny, so most of the list is those.
OUTLIER_COLUMNS = [
    "avg_monthly_credits",
    "avg_monthly_debits",
    "min_closing_balance",
    "avg_closing_balance",
    "avg_tx_count",
    "spend_housing_share",
    "spend_groceries_share",
    "spend_utilities_share",
    "spend_transport_share",
    "spend_shopping_share",
    "spend_medical_share",
    "spend_other_share",
    "expense_to_income_ratio",
    "income_cv",
    "balance_cv",
    "opening_balance_ratio",
    "liquidity_months_proxy",
    "transfer_in_ratio",
    "transfer_out_ratio",
    "avg_monthly_housing_payment",
]


def build_cashflow_features(applicants, transactions):
    app = applicants[["applicant_id", "application_date"]].copy()
    app["application_date"] = pd.to_datetime(app["application_date"])

    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    if "account_open_date" not in tx:
        tx["account_open_date"] = tx["transaction_date"]
    tx["account_open_date"] = pd.to_datetime(tx["account_open_date"])
    if "account_type" not in tx:
        tx["account_type"] = pd.NA

    tx = tx.merge(app, on="applicant_id", how="inner")
    tx = tx.loc[tx["transaction_date"] < tx["application_date"]].copy()
    tx = tx.sort_values(["applicant_id", "transaction_date"], kind="stable")

    tx["credit"] = tx["amount"].clip(lower=0)
    tx["debit"] = -tx["amount"].clip(upper=0)
    tx["month"] = tx["transaction_date"].dt.to_period("M")
    tx["is_income"] = tx["category"].isin(INCOME_CATEGORIES)
    tx["is_transfer"] = tx["category"].eq("TRANSFER")

    monthly = (
        tx.groupby(["applicant_id", "month"], observed=True)
        .agg(
            credits=("credit", "sum"),
            debits=("debit", "sum"),
            closing_balance=("balance_after", "last"),
            tx_count=("amount", "size"),
        )
        .reset_index()
    )

    base = (
        monthly.groupby("applicant_id")
        .agg(
            avg_monthly_credits=("credits", "mean"),
            std_monthly_credits=("credits", "std"),
            avg_monthly_debits=("debits", "mean"),
            avg_closing_balance=("closing_balance", "mean"),
            std_closing_balance=("closing_balance", "std"),
            min_closing_balance=("closing_balance", "min"),
            avg_tx_count=("tx_count", "mean"),
            observed_months=("month", "nunique"),
            total_credits=("credits", "sum"),
            total_debits=("debits", "sum"),
        )
        .reset_index()
    )

    # Opening balance is implied by the first transaction of the window.
    first = tx.groupby("applicant_id").head(1).set_index("applicant_id")
    base = base.merge(
        (first["balance_after"] - first["amount"]).rename("opening_balance"),
        left_on="applicant_id", right_index=True, how="left",
    )

    # Only salary and business income count as income. Transfers do not.
    income = tx.loc[tx["is_income"]]
    base = base.merge(
        income.groupby("applicant_id")
        .agg(income_total=("credit", "sum"), income_events=("credit", "size"))
        .reset_index(),
        on="applicant_id", how="left",
    )

    income_monthly = (
        income.groupby(["applicant_id", "month"], observed=True)["credit"]
        .sum()
        .reset_index()
    )
    base = base.merge(
        income_monthly.groupby("applicant_id")
        .agg(income_monthly_mean=("credit", "mean"),
             income_monthly_std=("credit", "std"))
        .reset_index(),
        on="applicant_id", how="left",
    )

    avg_monthly_income = (
        base["income_total"] / base["observed_months"].clip(lower=1)
    ).replace(0, np.nan)

    # Spend share per category, as a fraction of average monthly income.
    months = base["observed_months"].clip(lower=1)
    for bucket, category in SPEND_CATEGORIES.items():
        spend_total = (
            tx.loc[tx["category"].eq(category)].groupby("applicant_id")["debit"].sum()
        )
        base[f"spend_{bucket}_share"] = (
            base["applicant_id"].map(spend_total).fillna(0) / months / avg_monthly_income
        ).clip(0, 1).fillna(0)

    non_transfer = tx.loc[~tx["is_transfer"]]
    base = base.merge(
        non_transfer.groupby(["applicant_id", "month"], observed=True)
        .agg(non_transfer_credits=("credit", "sum"),
             non_transfer_debits=("debit", "sum"))
        .reset_index()
        .groupby("applicant_id")
        .agg(avg_monthly_non_transfer_credits=("non_transfer_credits", "mean"),
             avg_monthly_non_transfer_debits=("non_transfer_debits", "mean"))
        .reset_index(),
        on="applicant_id", how="left",
    )

    # Transfer reliance stays separate and scale-free.
    transfer = tx.loc[tx["is_transfer"]]
    transfer_stats = (
        transfer.groupby("applicant_id")
        .agg(transfer_credits=("credit", "sum"), transfer_debits=("debit", "sum"))
        .reset_index()
    )
    base = base.merge(transfer_stats, on="applicant_id", how="left")
    base[["transfer_credits", "transfer_debits"]] = (
        base[["transfer_credits", "transfer_debits"]].fillna(0)
    )
    base["transfer_in_ratio"] = base["transfer_credits"] / (base["total_credits"] + 1)
    base["transfer_out_ratio"] = base["transfer_debits"] / (base["total_debits"] + 1)

    base["income_cv"] = (
        base["income_monthly_std"] / (base["income_monthly_mean"].abs() + 1)
    ).replace([np.inf, -np.inf], np.nan)
    base["balance_cv"] = (
        base["std_closing_balance"] / (base["avg_closing_balance"].abs() + 1)
    ).replace([np.inf, -np.inf], np.nan)
    base["expense_to_income_ratio"] = (
        base["avg_monthly_non_transfer_debits"] / (avg_monthly_income + 1)
    )
    base["opening_balance_ratio"] = base["opening_balance"] / (avg_monthly_income + 1)
    base["liquidity_months_proxy"] = (
        base["min_closing_balance"] / (base["avg_monthly_non_transfer_debits"] + 1)
    )
    base["income_event_frequency"] = base["income_events"] / months

    app_dates = app.set_index("applicant_id")["application_date"]
    account_open = tx.groupby("applicant_id")["account_open_date"].min()
    tenure = (
        (app_dates.dt.year - account_open.dt.year) * 12
        + app_dates.dt.month - account_open.dt.month
    ).clip(lower=1).rename("account_tenure_months")
    base = base.merge(tenure.reset_index(), on="applicant_id", how="left")

    # Account type is one attribute per applicant, taken as the share of that
    # applicant's transactions on each account type.
    type_counts = (
        tx.groupby(["applicant_id", "account_type"], dropna=True)
        .size()
        .unstack(fill_value=0)
    )
    type_shares = type_counts.div(type_counts.sum(axis=1).replace(0, 1), axis=0)
    for account_type in ["CHECKING", "SAVINGS", "WALLET"]:
        column = f"account_type_{account_type.lower()}_share"
        if account_type in type_shares:
            type_shares[column] = type_shares[account_type]
        else:
            type_shares[column] = 0.0
    base = base.merge(
        type_shares[["account_type_checking_share",
                     "account_type_savings_share",
                     "account_type_wallet_share"]].reset_index(),
        on="applicant_id", how="left",
    )

    # Housing comes from the RENT category, not from generator metadata.
    rent = tx.loc[tx["category"].eq("RENT") & tx["debit"].gt(0)]
    rent_total = rent.groupby("applicant_id")["debit"].sum()
    base["has_housing_payment"] = base["applicant_id"].map(rent_total).notna().astype(int)
    base["avg_monthly_housing_payment"] = (
        base["applicant_id"].map(rent_total).fillna(0) / months
    )

    # Internal accounting columns, not features.
    base = base.drop(columns=[
        "income_total", "income_events",
        "income_monthly_mean", "income_monthly_std",
        "std_closing_balance",
        "avg_monthly_non_transfer_credits", "avg_monthly_non_transfer_debits",
        "total_credits", "total_debits",
        "transfer_credits", "transfer_debits",
    ])

    base = base.replace([np.inf, -np.inf], np.nan)
    numeric = base.select_dtypes(include=[np.number]).columns
    base[numeric] = base[numeric].fillna(0)
    return base.sort_values("applicant_id").reset_index(drop=True)


def feature_columns(features):
    return [c for c in features.columns if c != "applicant_id"]


def _iqr_bounds(features, train_ids, columns, multiplier):
    missing = [c for c in columns if c not in features.columns]
    if missing:
        raise ValueError(f"Columns not found in features: {missing}")

    train = features[features["applicant_id"].isin(train_ids)]
    bounds = {}
    for col in columns:
        q1, q3 = train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = {"lower": q1 - multiplier * iqr, "upper": q3 + multiplier * iqr}
    return bounds


def filter_extreme_applicants(features, train_ids, columns=None, multiplier=1.5):
    """Drop applicants outside train-fitted IQR bounds.

    Returns the filtered frame, the dropped rows (for an audit table) and the
    bounds that were used.
    """
    columns = columns or OUTLIER_COLUMNS
    bounds = _iqr_bounds(features, train_ids, columns, multiplier)

    extreme = pd.Series(False, index=features.index)
    for col, bound in bounds.items():
        extreme |= (features[col] < bound["lower"]) | (features[col] > bound["upper"])

    return features.loc[~extreme].copy(), features.loc[extreme].copy(), bounds


def winsorize_extreme_features(features, train_ids, columns=None, multiplier=1.5):
    """Clip features at train-fitted IQR bounds instead of dropping rows.

    Keeps every applicant, which matters for cohort coverage. Bounds are
    learned on training applicants only.
    """
    columns = columns or OUTLIER_COLUMNS
    bounds = _iqr_bounds(features, train_ids, columns, multiplier)

    winsorized = features.copy()
    clipped_counts = {}
    for col, bound in bounds.items():
        clipped = features[col].lt(bound["lower"]) | features[col].gt(bound["upper"])
        clipped_counts[col] = int(clipped.sum())
        winsorized[col] = features[col].clip(lower=bound["lower"], upper=bound["upper"])

    return winsorized, bounds, clipped_counts
