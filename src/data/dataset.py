"""Build the modelling dataset: synthetic book, cash-flow features, vintage splits."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_data import GENERATOR_ONLY_COLUMNS, generate_data
from src.features.cashflow_features import build_cashflow_features

TARGET = "default_12m"


def temporal_split(applicants, valid_fraction=.15, test_fraction=.20):
    """Split on whole application vintages so no vintage lands in two cohorts."""
    a = applicants.copy()
    a["application_date"] = pd.to_datetime(a["application_date"])
    dates = np.array(sorted(a["application_date"].dropna().unique()))

    n_dates = len(dates)
    n_test = max(1, int(round(n_dates * test_fraction)))
    n_valid = max(1, int(round(n_dates * valid_fraction)))
    n_train = n_dates - n_valid - n_test
    if n_train < 1:
        raise ValueError("Not enough distinct application vintages for temporal split.")

    train_dates = dates[:n_train]
    valid_dates = dates[n_train:n_train + n_valid]

    train = a[a["application_date"].isin(train_dates)].copy()
    valid = a[a["application_date"].isin(valid_dates)].copy()
    test = a[a["application_date"].isin(dates[n_train + n_valid:])].copy()

    assert train["application_date"].max() < valid["application_date"].min()
    assert valid["application_date"].max() < test["application_date"].min()
    return train, valid, test


def temporal_split_with_calibration(applicants, valid_fraction=.15,
                                    calibration_fraction=.05, test_fraction=.20):
    if min(valid_fraction, calibration_fraction, test_fraction) < 0:
        raise ValueError("Split fractions must be non-negative.")
    if valid_fraction + calibration_fraction + test_fraction >= 1:
        raise ValueError(
            "Validation, calibration, and test fractions must sum to less than one."
        )

    a = applicants.copy()
    a["application_date"] = pd.to_datetime(a["application_date"])
    dates = np.array(sorted(a["application_date"].dropna().unique()))

    n_dates = len(dates)
    n_test = max(1, int(round(n_dates * test_fraction)))
    n_calibration = max(1, int(round(n_dates * calibration_fraction)))
    n_valid = max(1, int(round(n_dates * valid_fraction)))
    n_train = n_dates - n_test - n_calibration - n_valid
    if n_train < 1:
        raise ValueError("Not enough distinct application vintages for four-way split.")

    train_end = n_train
    valid_end = train_end + n_valid
    calibration_end = valid_end + n_calibration

    train = a[a["application_date"].isin(dates[:train_end])].copy()
    valid = a[a["application_date"].isin(dates[train_end:valid_end])].copy()
    calibration = a[a["application_date"].isin(dates[valid_end:calibration_end])].copy()
    test = a[a["application_date"].isin(dates[calibration_end:])].copy()

    assert train["application_date"].max() < valid["application_date"].min()
    assert valid["application_date"].max() < calibration["application_date"].min()
    assert calibration["application_date"].max() < test["application_date"].min()
    return train, valid, calibration, test


def build_dataset(cfg, seed, n_override=None):
    """Generate the book, featurise it and split by application vintage.

    Writes the raw and derived tables under ``cfg["data"]["output_dir"]`` and
    returns the modelling frame, the transactions, the feature frame, the four
    split cohorts and the simulator oracle probabilities.
    """
    data_cfg = cfg["data"]
    applicants, transactions = generate_data(
        n_applicants=n_override or data_cfg["n_applicants"],
        months_history=data_cfg["months_history"],
        start_application_date=data_cfg["start_application_date"],
        end_application_date=data_cfg["end_application_date"],
        seed=seed,
        variable_history=bool(data_cfg.get("variable_history", False)),
        min_history_months=data_cfg.get("min_history_months"),
        maturity_gap_months=int(data_cfg.get("maturity_gap_months", 0)),
        drift_start_application_date=data_cfg.get("drift_start_application_date"),
        drift_income_multiplier=float(data_cfg.get("drift_income_multiplier", 1.0)),
        drift_obligation_delta=float(data_cfg.get("drift_obligation_delta", 0.0)),
        base_default_logit=float(data_cfg.get("base_default_logit", -2.47)),
    )

    features = build_cashflow_features(applicants, transactions)
    data_out = Path(data_cfg["output_dir"])
    data_out.mkdir(parents=True, exist_ok=True)
    applicants.to_parquet(data_out / "applicants.parquet", index=False)
    transactions.to_parquet(data_out / "transactions.parquet", index=False)
    features.to_parquet(data_out / "cashflow_features.parquet", index=False)

    # The oracle uses the probability before the noise draw, so it measures what
    # the simulated drivers know rather than the realised outcome.
    oracle_by_id = applicants.set_index("applicant_id")["p_default_conditional"]

    # Modelling table: ids, timing and target from the applicant frame, all
    # features from the cash-flow builder. Nothing else gets in.
    app = applicants[["applicant_id", "application_date", TARGET]].merge(
        features, on="applicant_id", how="left", validate="one_to_one"
    )
    # Should be a no-op here, kept so a notebook path cannot leak latents either.
    app = app.drop(columns=GENERATOR_ONLY_COLUMNS, errors="ignore")

    exp_cfg = cfg["experiment"]
    train, valid, calibration, test = temporal_split_with_calibration(
        app,
        exp_cfg["validation_fraction_by_time"],
        exp_cfg["calibration_fraction_by_time"],
        exp_cfg["test_fraction_by_time"],
    )

    split_assignments = pd.concat([
        train[["applicant_id"]].assign(split="train"),
        valid[["applicant_id"]].assign(split="validation"),
        calibration[["applicant_id"]].assign(split="calibration"),
        test[["applicant_id"]].assign(split="test"),
    ], ignore_index=True)
    app.to_parquet(data_out / "applicant_modeling.parquet", index=False)
    split_assignments.to_parquet(data_out / "split_assignments.parquet", index=False)

    return {
        "app": app,
        "transactions": transactions,
        "features": features,
        "train": train,
        "validation": valid,
        "calibration": calibration,
        "test": test,
        "oracle": oracle_by_id,
    }
