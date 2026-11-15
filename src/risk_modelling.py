"""Supervised stage: baseline vs hybrid models, tuning, calibration and stability."""
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.dataset import TARGET
from src.evaluation.metrics import evaluate, psi
from src.models.factory import make_lgbm
from src.models.risk_model import fit_and_evaluate


def _logit(probability):
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_feature_sets(train, valid, test, feature_sets, lgbm_cfg, seed, out_dir,
                     label_fraction=1.0):
    """Fit every model family on every feature set, sharing one labelled sample.

    ``feature_sets`` is a list of (directory name, label, columns). Fitted
    pipelines go to ``out_dir/<directory name>_models`` and the label ends up in
    the feature_set column of the returned metrics.
    """
    if label_fraction < 1:
        strat = train[TARGET] if train[TARGET].nunique() > 1 else None
        train_labeled, _ = train_test_split(
            train, train_size=label_fraction, random_state=seed, stratify=strat
        )
    else:
        train_labeled = train

    frames = []
    models = {}
    for dir_name, label, columns in feature_sets:
        metrics, fitted = fit_and_evaluate(
            train_labeled[columns], train_labeled[TARGET],
            valid[columns], valid[TARGET],
            test[columns], test[TARGET],
            out_dir / f"{dir_name}_models", lgbm_cfg, seed,
        )
        metrics["feature_set"] = label
        frames.append(metrics)
        models[dir_name] = fitted

    return pd.concat(frames, ignore_index=True), models, train_labeled


def tune_feature_sets(train_labeled, valid, test, feature_sets, lgbm_cfg, grid, seed):
    """Small validation-AUC grid per feature set. Test is only scored at the end."""
    results = {}
    for _dir_name, label, columns in feature_sets:
        best_score = -np.inf
        best_params = None
        best_model = None

        for n_estimators in grid["n_estimators"]:
            for num_leaves in grid["num_leaves"]:
                params = {
                    **lgbm_cfg,
                    "n_estimators": int(n_estimators),
                    "num_leaves": int(num_leaves),
                }
                candidate = make_lgbm(params, seed)
                candidate.fit(train_labeled[columns], train_labeled[TARGET])
                prob = candidate.predict_proba(valid[columns])[:, 1]
                auc = evaluate(valid[TARGET], prob)["roc_auc"]
                score = auc if np.isfinite(auc) else -1.0
                if score > best_score:
                    best_score, best_params, best_model = score, params, candidate

        test_prob = best_model.predict_proba(test[columns])[:, 1]
        results[label] = {
            "selected_params": best_params,
            "validation_roc_auc": float(best_score) if best_score >= 0 else None,
            "test_metrics": evaluate(test[TARGET], test_prob),
        }
    return results


def oracle_ceiling(test, oracle_by_id, test_ids):
    """How well the simulator's own conditional probability ranks the test cohort."""
    return {
        "score_source": (
            "generator conditional default probability; simulator-only oracle"
        ),
        "test_metrics": evaluate(
            test[TARGET], oracle_by_id.loc[test_ids].to_numpy()
        ),
    }


def calibrate_hybrid(model, calibration, test, hybrid_cols, calibrator_path):
    """Platt scaling fitted on the calibration vintage, reported on test."""
    probabilities = model.predict_proba(calibration[hybrid_cols])[:, 1]

    if calibration[TARGET].nunique() < 2:
        return {
            "method": "Platt scaling",
            "status": "skipped_insufficient_classes",
            "calibration_n": int(len(calibration)),
        }

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(_logit(probabilities), calibration[TARGET])
    dump(calibrator, calibrator_path)

    test_prob = model.predict_proba(test[hybrid_cols])[:, 1]
    calibrated = calibrator.predict_proba(_logit(test_prob))[:, 1]
    return {
        "method": "Platt scaling",
        "status": "fit",
        "calibration_n": int(len(calibration)),
        "test_metrics": evaluate(test[TARGET], calibrated),
    }


def label_efficiency(train, test, feature_sets, fractions, lgbm_cfg, seed):
    """How much the hybrid needs to match the baseline when labels are scarce."""
    rows = []
    for fraction in fractions:
        if fraction >= 1:
            subset = train
        else:
            strat = train[TARGET] if train[TARGET].nunique() > 1 else None
            subset, _ = train_test_split(
                train, train_size=fraction, random_state=seed, stratify=strat
            )
        for label, columns in feature_sets:
            pipe = make_lgbm(lgbm_cfg, seed)
            pipe.fit(subset[columns], subset[TARGET])
            prob = pipe.predict_proba(test[columns])[:, 1]
            rows.append({
                "label_fraction": fraction,
                "feature_set": label,
                **evaluate(test[TARGET], prob),
            })
    return pd.DataFrame(rows)


def score_stability(model, valid, test, hybrid_cols):
    """PSI of the saved hybrid model's validation scores against its test scores."""
    p_valid = model.predict_proba(valid[hybrid_cols])[:, 1]
    p_test = model.predict_proba(test[hybrid_cols])[:, 1]
    return psi(p_valid, p_test)
