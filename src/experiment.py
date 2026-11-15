"""End-to-end run: build the dataset, learn representations, fit risk models."""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import yaml
from joblib import load

from src.data.dataset import TARGET, build_dataset
from src.evaluation.embedding_diagnostics import embedding_activity_diagnostics
from src.features.cashflow_features import feature_columns
from src.models.pretrain import select_device
from src.representation_learning import embed_applicants, pretrain_encoder
from src.risk_modelling import (
    calibrate_hybrid,
    fit_feature_sets,
    label_efficiency,
    oracle_ceiling,
    score_stability,
    tune_feature_sets,
)

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "torch", "pyarrow"]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def _git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    found = {}
    for package in PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def main(config_path="config.yaml", n_override=None, epochs_override=None,
         model_seed_override=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg["seed"])  # data generation seed
    model_seed = seed if model_seed_override is None else int(model_seed_override)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    out = Path(cfg["experiment"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    exp_cfg = cfg["experiment"]

    # 1. data
    dataset = build_dataset(cfg, seed, n_override)
    app = dataset["app"]
    transactions = dataset["transactions"]
    features = dataset["features"]
    train, valid = dataset["train"], dataset["validation"]
    calibration, test = dataset["calibration"], dataset["test"]
    oracle_by_id = dataset["oracle"]

    train_ids = train["applicant_id"].astype(int).tolist()
    valid_ids = valid["applicant_id"].astype(int).tolist()
    calibration_ids = calibration["applicant_id"].astype(int).tolist()
    test_ids = test["applicant_id"].astype(int).tolist()

    # 2. transaction representations, trained on training vintages only
    model, pretrain_audit = pretrain_encoder(
        app, transactions, train_ids, cfg, model_seed, out, epochs=epochs_override
    )
    pretrain_ids = pretrain_audit["pretrain_ids"]
    n_valid_leak = len(pretrain_ids & set(valid_ids))
    n_calibration_leak = len(pretrain_ids & set(calibration_ids))
    n_test_leak = len(pretrain_ids & set(test_ids))
    assert n_valid_leak == n_calibration_leak == n_test_leak == 0

    max_length = int(cfg["model"]["max_sequence_length"])
    embeddings, emb_cols = embed_applicants(
        model, app, transactions,
        train_ids + valid_ids + calibration_ids + test_ids, max_length,
    )
    embeddings.to_parquet(Path(cfg["data"]["output_dir"]) / "finrep_embeddings.parquet",
                          index=False)
    _write_json(
        out / "embedding_diagnostics.json",
        embedding_activity_diagnostics(embeddings, features, train_ids),
    )

    app = app.merge(embeddings, on="applicant_id", how="left")
    train = app[app["applicant_id"].isin(train_ids)].copy()
    valid = app[app["applicant_id"].isin(valid_ids)].copy()
    calibration = app[app["applicant_id"].isin(calibration_ids)].copy()
    test = app[app["applicant_id"].isin(test_ids)].copy()

    # 3. supervised models: baseline vs hybrid on the same labelled applicants
    base_cols = feature_columns(features)
    assert not {TARGET, "application_date"} & set(base_cols)
    hybrid_cols = base_cols + emb_cols

    feature_sets = [
        ("baseline", "cashflow", base_cols),
        ("hybrid", "cashflow_plus_finrep", hybrid_cols),
    ]
    metrics, models, train_labeled = fit_feature_sets(
        train, valid, test, feature_sets, cfg["lgbm"], model_seed, out,
        label_fraction=float(exp_cfg["primary_label_fraction"]),
    )
    metrics.to_csv(out / "model_metrics.csv", index=False)

    strong_results = tune_feature_sets(
        train_labeled, valid, test, feature_sets, cfg["lgbm"],
        exp_cfg["strong_baseline_grid"], model_seed,
    )
    _write_json(out / "strong_baseline_metrics.json", strong_results)
    _write_json(out / "oracle_ceiling_metrics.json",
                oracle_ceiling(test, oracle_by_id, test_ids))

    # 4. calibration on the held-out vintage, reported on test
    calibration_report = calibrate_hybrid(
        models["hybrid"]["lightgbm"], calibration, test, hybrid_cols,
        out / "hybrid_models" / "platt_calibrator.joblib",
    )
    _write_json(out / "calibration_metrics.json", calibration_report)

    # 5. label efficiency and score stability
    efficiency = label_efficiency(
        train, test,
        [("cashflow", base_cols), ("hybrid", hybrid_cols)],
        exp_cfg["label_fractions"], cfg["lgbm"], model_seed,
    )
    efficiency.to_csv(out / "label_efficiency.csv", index=False)

    saved_hybrid = load(out / "hybrid_models" / "lightgbm.joblib")
    _write_json(out / "score_psi.json", {
        "validation_vs_test_score_psi": score_stability(
            saved_hybrid, valid, test, hybrid_cols
        ),
    })

    # 6. split audit and run manifest
    split_summary = {
        "train_n": len(train), "validation_n": len(valid),
        "calibration_n": len(calibration), "test_n": len(test),
        "train_date_min": str(train["application_date"].min().date()),
        "train_date_max": str(train["application_date"].max().date()),
        "validation_date_min": str(valid["application_date"].min().date()),
        "validation_date_max": str(valid["application_date"].max().date()),
        "calibration_date_min": str(calibration["application_date"].min().date()),
        "calibration_date_max": str(calibration["application_date"].max().date()),
        "test_date_min": str(test["application_date"].min().date()),
        "test_date_max": str(test["application_date"].max().date()),
        "train_labelled_n": len(train_labeled),
        "train_default_rate": float(train[TARGET].mean()),
        "validation_default_rate": float(valid[TARGET].mean()),
        "test_default_rate": float(test[TARGET].mean()),
        "pretraining_applicant_n": len(pretrain_ids),
        "pretraining_fit_applicant_n": len(pretrain_audit["fit_ids"]),
        "pretraining_validation_applicant_n": len(pretrain_audit["validation_ids"]),
        "validation_applicants_used_in_pretraining": n_valid_leak,
        "calibration_applicants_used_in_pretraining": n_calibration_leak,
        "test_applicants_used_in_pretraining": n_test_leak,
    }
    _write_json(out / "split_summary.json", split_summary)

    _write_json(out / "run_manifest.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "data_generation_seed": seed,
        "model_seed": model_seed,
        "git_hash": _git_hash(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "device": str(select_device()),
    })

    print(metrics.to_string(index=False))
    print(f"\nArtifacts written to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-applicants", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    main(args.config, args.n_applicants, args.epochs)
