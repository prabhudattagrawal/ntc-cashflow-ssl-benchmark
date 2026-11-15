"""Approval policy: AUC is a whole-ranking number, a decision is a cutoff.

At a fixed approval rate only the applicants near the cutoff matter, so this
module works directly from scores and turns them into defaults avoided and
expected loss.
"""
import numpy as np
import pandas as pd

DEFAULT_EAD = 10_000.0
DEFAULT_LGD = 0.60


def cutoff_at_approval_rate(scores, approval_rate):
    """Cutoff that approves the safest ``approval_rate`` of the book.

    Scores are PDs, so lower is safer and approve means score <= cutoff.
    """
    scores = np.asarray(scores, dtype=float)
    if not 0.0 < approval_rate < 1.0:
        raise ValueError("approval_rate must be strictly between 0 and 1")
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    return float(np.quantile(scores, approval_rate))


def approval_policy_table(y_true, scores_by_model, approval_rates,
                          ead=DEFAULT_EAD, lgd=DEFAULT_LGD):
    y = np.asarray(y_true, dtype=int)
    total_defaults = int(y.sum())
    if total_defaults == 0:
        raise ValueError("y_true contains no defaults")

    rows = []
    for model_name, scores in scores_by_model.items():
        scores = np.asarray(scores, dtype=float)
        if scores.shape[0] != y.shape[0]:
            raise ValueError(
                f"score length {scores.shape[0]} does not match "
                f"labels {y.shape[0]}"
            )

        for rate in approval_rates:
            cutoff = cutoff_at_approval_rate(scores, float(rate))
            approved = scores <= cutoff
            n_approved = int(approved.sum())
            approved_defaults = int(y[approved].sum())
            rejected_defaults = total_defaults - approved_defaults
            bad_rate = approved_defaults / n_approved

            rows.append({
                "model": model_name,
                "approval_rate": float(rate),
                "cutoff": cutoff,
                "approved_n": n_approved,
                "approved_defaults": approved_defaults,
                "approved_default_rate": bad_rate,
                "defaults_per_10k_approved": bad_rate * 10_000,
                "rejected_defaults": rejected_defaults,
                "default_capture_rate": rejected_defaults / total_defaults,
                "expected_loss": approved_defaults * ead * lgd,
                "loss_per_10k_approved": bad_rate * 10_000 * ead * lgd,
                "loss_per_10k_applicants": bad_rate * 10_000 * ead * lgd * float(rate),
            })
    return pd.DataFrame(rows)


def approval_policy_deltas(table, baseline="cashflow",
                           challenger="cashflow_plus_finrep"):
    """Challenger minus baseline at each shared approval rate.

    Positive defaults_avoided_per_10k_approved means a cleaner book at the
    same volume.
    """
    base = table.loc[table["model"].eq(baseline)].set_index("approval_rate")
    chal = table.loc[table["model"].eq(challenger)].set_index("approval_rate")
    shared = base.index.intersection(chal.index)
    if shared.empty:
        raise ValueError("baseline and challenger share no approval rates")

    base_bad = base.loc[shared, "approved_default_rate"]
    chal_bad = chal.loc[shared, "approved_default_rate"]
    base_capture = base.loc[shared, "default_capture_rate"]
    chal_capture = chal.loc[shared, "default_capture_rate"]

    out = pd.DataFrame(index=shared)
    out["approved_n"] = base.loc[shared, "approved_n"]
    out["baseline_bad_rate"] = base_bad
    out["challenger_bad_rate"] = chal_bad
    out["delta_bad_rate_pp"] = (chal_bad - base_bad) * 100.0
    out["defaults_avoided_per_10k_approved"] = (
        base.loc[shared, "defaults_per_10k_approved"]
        - chal.loc[shared, "defaults_per_10k_approved"]
    )
    out["loss_avoided_per_10k_approved"] = (
        base.loc[shared, "loss_per_10k_approved"]
        - chal.loc[shared, "loss_per_10k_approved"]
    )
    out["loss_avoided_per_10k_applicants"] = (
        out["loss_avoided_per_10k_approved"] * out.index
    )
    out["relative_loss_reduction"] = (
        out["defaults_avoided_per_10k_approved"]
        / base.loc[shared, "defaults_per_10k_approved"]
    )
    out["baseline_capture"] = base_capture
    out["challenger_capture"] = chal_capture
    out["capture_delta_pp"] = (chal_capture - base_capture) * 100.0
    return out.reset_index().rename(columns={"index": "approval_rate"})


def _defaults_avoided_per_10k(y, baseline, challenger, rate):
    base_cutoff = cutoff_at_approval_rate(baseline, rate)
    chal_cutoff = cutoff_at_approval_rate(challenger, rate)
    base_bad = y[baseline <= base_cutoff].mean()
    chal_bad = y[challenger <= chal_cutoff].mean()
    return (base_bad - chal_bad) * 10_000


def bootstrap_policy_delta(y_true, scores_baseline, scores_challenger,
                           approval_rates, n_boot=1000, seed=42,
                           ead=DEFAULT_EAD, lgd=DEFAULT_LGD):
    """Paired bootstrap for defaults avoided per 10k approved.

    Both cutoffs are recomputed inside each replicate, so the interval carries
    ranking uncertainty and not just label noise.
    """
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(scores_baseline, dtype=float)
    chal = np.asarray(scores_challenger, dtype=float)
    n = y.shape[0]
    rates = [float(r) for r in approval_rates]
    rng = np.random.default_rng(seed)

    point = {rate: _defaults_avoided_per_10k(y, base, chal, rate) for rate in rates}
    draws = np.empty((n_boot, len(rates)))

    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for j, rate in enumerate(rates):
            draws[b, j] = _defaults_avoided_per_10k(y[idx], base[idx], chal[idx], rate)

    rows = []
    for j, rate in enumerate(rates):
        values = draws[:, j]
        rows.append({
            "approval_rate": rate,
            "defaults_avoided_per_10k_approved": point[rate],
            "ci_2_5": float(np.percentile(values, 2.5)),
            "ci_97_5": float(np.percentile(values, 97.5)),
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "p_nonpositive": float(np.mean(values <= 0)),
            "loss_avoided_per_10k_approved": point[rate] * ead * lgd,
            "loss_avoided_per_10k_applicants": point[rate] * ead * lgd * rate,
            "n_boot_valid": int(values.shape[0]),
        })
    return pd.DataFrame(rows)


def swap_analysis(y_true, scores_baseline, scores_challenger, approval_rate):
    """Who flips decision when the challenger replaces the baseline."""
    y = np.asarray(y_true, dtype=int)
    base = np.asarray(scores_baseline, dtype=float)
    chal = np.asarray(scores_challenger, dtype=float)

    base_approved = base <= cutoff_at_approval_rate(base, approval_rate)
    chal_approved = chal <= cutoff_at_approval_rate(chal, approval_rate)

    # At a fixed approval rate the two swap groups have the same size.
    groups = {
        "swapped_out": base_approved & ~chal_approved,
        "swapped_in": ~base_approved & chal_approved,
    }

    rows = []
    for name, mask in groups.items():
        n = int(mask.sum())
        rows.append({
            "group": name,
            "applicants": n,
            "defaults": int(y[mask].sum()),
            "default_rate": float(y[mask].mean()) if n else np.nan,
        })
    return pd.DataFrame(rows)


def fixed_cutoff_comparison(y_true, scores_by_model, cutoff,
                            ead=DEFAULT_EAD, lgd=DEFAULT_LGD):
    """Fixed threshold, floating approval rate. The deployed scorecard view."""
    y = np.asarray(y_true, dtype=int)
    rows = []
    for model_name, scores in scores_by_model.items():
        scores = np.asarray(scores, dtype=float)
        approved = scores <= cutoff
        n = int(approved.sum())
        defaults = int(y[approved].sum())
        rows.append({
            "model": model_name,
            "cutoff": cutoff,
            "approved_n": n,
            "approval_rate": n / scores.shape[0],
            "approved_default_rate": defaults / n if n else np.nan,
            "defaults_per_10k_approved": (defaults / n * 10_000) if n else np.nan,
            "expected_loss": defaults * ead * lgd,
        })
    return pd.DataFrame(rows)


def capture_curve(y_true, scores, n_points=200):
    """Cumulative share of defaults caught as the riskiest are rejected."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    total = int(y_sorted.sum())
    if total == 0:
        raise ValueError("y_true contains no defaults")

    share_rejected = np.arange(1, y_sorted.size + 1) / y_sorted.size
    share_captured = np.cumsum(y_sorted) / total
    idx = np.unique(
        np.linspace(0, y_sorted.size - 1, min(n_points, y_sorted.size)).astype(int)
    )
    return pd.DataFrame({
        "share_rejected": share_rejected[idx],
        "share_defaults_captured": share_captured[idx],
    })
