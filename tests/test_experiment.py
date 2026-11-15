import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.embedding_diagnostics import embedding_activity_diagnostics
from src.experiment import main


def test_embedding_diagnostics_pick_up_activity():
    activity = list(range(20))
    features = pd.DataFrame({
        "applicant_id": range(20),
        "avg_tx_count": activity,
        "observed_months": [12] * 20,
        "avg_monthly_credits": [value * 10 for value in activity],
        "avg_monthly_debits": [value * 5 for value in activity],
        "income_event_frequency": [value / 2 for value in activity],
        "expense_to_income_ratio": [0.5] * 20,
    })
    embeddings = pd.DataFrame({
        "applicant_id": range(20),
        "finrep_000": activity,
        "finrep_001": [float(value % 3) for value in activity],
        "finrep_002": [1.0] * 20,
    })

    report = embedding_activity_diagnostics(embeddings, features, list(range(20)))

    assert report["scope"] == "training_vintages_only"
    assert report["max_abs_spearman_by_activity_feature"]["avg_tx_count"] == 1.0
    assert report["near_constant_embedding_fraction"] > 0


def test_smoke_run_writes_manifest_and_audits(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "config.yaml").read_text())
    config["data"]["output_dir"] = str(tmp_path / "data")
    config["experiment"]["output_dir"] = str(tmp_path / "artifacts")
    config["experiment"]["label_fractions"] = [1.0]
    config["experiment"]["strong_baseline_grid"] = {
        "n_estimators": [5],
        "num_leaves": [7],
    }
    config_path = tmp_path / "smoke_config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    main(config_path=config_path, n_override=300, epochs_override=1)

    artifacts = tmp_path / "artifacts"
    manifest = json.loads((artifacts / "run_manifest.json").read_text())
    split_summary = json.loads((artifacts / "split_summary.json").read_text())

    assert manifest["config"]["data"]["n_applicants"] == 48000
    assert manifest["device"]
    assert split_summary["validation_applicants_used_in_pretraining"] == 0
    assert split_summary["calibration_applicants_used_in_pretraining"] == 0
    assert split_summary["test_applicants_used_in_pretraining"] == 0
    assert (artifacts / "oracle_ceiling_metrics.json").exists()
    assert (artifacts / "calibration_metrics.json").exists()
    assert (artifacts / "embedding_diagnostics.json").exists()

    modeling_path = tmp_path / "data" / "applicant_modeling.parquet"
    assert modeling_path.exists()
    model_table = pd.read_parquet(modeling_path)

    # None of the generator's hidden columns may reach the modelling table.
    generator_only = [
        "profile",
        "base_income",
        "default_probability",
        "p_default_conditional",
        "p_default_realized",
        "sequence_stress_latent",
        "obligation_latent",
    ]
    for column in generator_only:
        assert column not in model_table.columns
