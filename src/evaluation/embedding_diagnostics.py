import pandas as pd

ACTIVITY_COLUMNS = [
    "avg_tx_count",
    "observed_months",
    "avg_monthly_credits",
    "avg_monthly_debits",
    "income_event_frequency",
    "expense_to_income_ratio",
]


def embedding_activity_diagnostics(embeddings, features, applicant_ids):
    """Sanity check that the finrep dims are not just re-learning volume."""
    emb_cols = [c for c in embeddings if c.startswith("finrep_")]
    activity_cols = [c for c in ACTIVITY_COLUMNS if c in features.columns]
    if not emb_cols or not activity_cols:
        raise ValueError("Embedding and cash-flow activity features are required.")

    emb = embeddings.set_index("applicant_id").loc[applicant_ids, emb_cols]
    act = features.set_index("applicant_id").loc[applicant_ids, activity_cols]

    corr = pd.DataFrame(index=emb_cols, columns=activity_cols, dtype=float)
    for e_col in emb_cols:
        if emb[e_col].nunique(dropna=True) < 2:
            continue
        for a_col in activity_cols:
            if act[a_col].nunique(dropna=True) < 2:
                continue
            corr.loc[e_col, a_col] = emb[e_col].corr(act[a_col], method="spearman")

    abs_corr = corr.abs()
    strongest = abs_corr.max(axis=1)

    top = {}
    for e_col in emb_cols:
        if pd.isna(strongest[e_col]):
            top[e_col] = {"activity_feature": None, "absolute_spearman": None}
        else:
            top[e_col] = {
                "activity_feature": str(abs_corr.loc[e_col].idxmax()),
                "absolute_spearman": float(strongest[e_col]),
            }

    std = emb.std(axis=0)
    return {
        "scope": "training_vintages_only",
        "applicant_n": int(len(applicant_ids)),
        "embedding_dimension_n": int(len(emb_cols)),
        "activity_features": activity_cols,
        "embedding_std_min": float(std.min()),
        "embedding_std_median": float(std.median()),
        "near_constant_embedding_fraction": float((std < 1e-8).mean()),
        "mean_abs_spearman_by_activity_feature": {
            c: float(abs_corr[c].mean()) for c in activity_cols
        },
        "max_abs_spearman_by_activity_feature": {
            c: float(abs_corr[c].max()) for c in activity_cols
        },
        # The paper quotes this one: a dim that tracks activity at 0.95+ is
        # probably an activity proxy wearing a transformer costume.
        "strongly_activity_correlated_embedding_fraction": float(
            (strongest >= 0.95).mean()
        ),
        "top_correlated_activity_by_embedding": top,
    }
