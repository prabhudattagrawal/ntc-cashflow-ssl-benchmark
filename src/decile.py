import numpy as np
import pandas as pd


def _capture_table(y_true, y_prob, n_bins, bin_name):
    df = pd.DataFrame({
        "actual_default": np.asarray(y_true),
        "predicted_pd": np.asarray(y_prob),
    })

    # Rank before binning, otherwise qcut puts all the tied scores in one bin.
    df[bin_name] = pd.qcut(
        df["predicted_pd"].rank(method="first"), q=n_bins, labels=False
    ) + 1

    table = (
        df.groupby(bin_name, observed=True)
        .agg(
            applicants=("actual_default", "size"),
            defaults=("actual_default", "sum"),
            avg_predicted_pd=("predicted_pd", "mean"),
            actual_default_rate=("actual_default", "mean"),
        )
        .reset_index()
    )

    total_defaults = df["actual_default"].sum()
    table["default_capture_pct"] = table["defaults"] / total_defaults
    table["cumulative_default_capture_pct"] = table["default_capture_pct"].cumsum()
    table["lift"] = table["actual_default_rate"] / df["actual_default"].mean()

    pct_cols = [
        "avg_predicted_pd",
        "actual_default_rate",
        "default_capture_pct",
        "cumulative_default_capture_pct",
    ]
    table[pct_cols] = table[pct_cols] * 100
    return table.round(2)


def decile_analysis(y_true, y_prob, n_bins=10):
    """Default rate, capture and lift by predicted-PD decile."""
    return _capture_table(y_true, y_prob, n_bins, "decile")


def pentile_analysis(y_true, y_prob, n_bins=5):
    return _capture_table(y_true, y_prob, n_bins, "pentile")
