import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.sequences import MASK_ID, PAD_ID
from src.models.pretrain import pretrain
from src.models.risk_model import fit_and_evaluate
from src.models.transaction_encoder import TransactionEncoder, masked_category_loss


def test_encoder_output_shapes():
    model = TransactionEncoder(embedding_dim=16, attention_heads=4, layers=1, max_length=4)
    categories = torch.tensor([[0, 3, PAD_ID, PAD_ID], [1, 2, 4, PAD_ID]])
    padding = categories.eq(PAD_ID)

    output = model(
        categories,
        torch.zeros_like(categories, dtype=torch.float32),  # amount
        torch.zeros_like(categories, dtype=torch.float32),  # relative_amount
        torch.zeros_like(categories),                      # time
        torch.zeros_like(categories),                      # direction
        padding,
    )

    assert output["embedding"].shape == (2, 16)
    assert output["category_logits"].shape == (2, 4, 10)


def test_padding_is_ignored():
    torch.manual_seed(7)
    model = TransactionEncoder(
        embedding_dim=16, attention_heads=4, layers=1, max_length=4
    ).eval()
    categories = torch.tensor([[0, 3, PAD_ID, PAD_ID]])
    padding = categories.eq(PAD_ID)

    first = model(
        categories,
        torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
        torch.tensor([[1, 2, 0, 0]]),
        torch.tensor([[1, 0, 0, 0]]),
        padding,
    )["embedding"]

    # Same real tokens, garbage in the padded slots.
    with_garbage = model(
        torch.tensor([[0, 3, 7, 8]]),
        torch.tensor([[1.0, 2.0, 11.0, 12.0]]),
        torch.tensor([[1.0, 2.0, 13.0, 14.0]]),
        torch.tensor([[1, 2, 13, 14]]),
        torch.tensor([[1, 0, 1, 1]]),
        padding,
    )["embedding"]

    torch.testing.assert_close(first, with_garbage)


def test_masking_leaves_padding_alone():
    class CaptureModel(nn.Module):
        def forward(self, category, amount, relative_amount, time, direction, padding):
            self.seen_categories = category.detach().clone()
            logits = torch.zeros((*category.shape, 10), requires_grad=True)
            return {"category_logits": logits}

    model = CaptureModel()
    categories = torch.tensor([[0, 1, PAD_ID, PAD_ID]])
    batch = {
        "category": categories,
        "amount": torch.zeros_like(categories),
        "relative_amount": torch.zeros_like(categories, dtype=torch.float32),
        "time": torch.zeros_like(categories),
        "direction": torch.zeros_like(categories),
        "padding": categories.eq(PAD_ID),
    }

    loss = masked_category_loss(model, batch, mask_probability=1.0)

    assert torch.isfinite(loss)
    assert model.seen_categories[0, :2].tolist() == [MASK_ID, MASK_ID]
    assert model.seen_categories[0, 2:].tolist() == [PAD_ID, PAD_ID]


def test_saved_pipeline_reproduces_predictions(tmp_path):
    rng = np.random.default_rng(42)
    features = pd.DataFrame({
        "signal": rng.normal(size=120),
        "noise": rng.normal(size=120),
    })
    target = pd.Series((features["signal"] > 0).astype(int))
    params = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 5,
        "min_child_samples": 5,
    }

    _, models = fit_and_evaluate(
        features.iloc[:80], target.iloc[:80],
        features.iloc[80:100], target.iloc[80:100],
        features.iloc[100:], target.iloc[100:],
        tmp_path, params, seed=42,
    )

    fitted = models["lightgbm"].predict_proba(features.iloc[100:])[:, 1]
    reloaded = joblib.load(tmp_path / "lightgbm.joblib")
    saved = reloaded.predict_proba(features.iloc[100:])[:, 1]

    np.testing.assert_array_equal(fitted, saved)


def test_models_honour_seed(tmp_path):
    rng = np.random.default_rng(11)
    features = pd.DataFrame({
        "signal": rng.normal(size=90),
        "noise": rng.normal(size=90),
    })
    target = pd.Series((features["signal"] > 0).astype(int))
    params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 5,
        "min_child_samples": 5,
    }

    _, models = fit_and_evaluate(
        features.iloc[:60], target.iloc[:60],
        features.iloc[60:75], target.iloc[60:75],
        features.iloc[75:], target.iloc[75:],
        tmp_path, params, seed=7,
    )

    assert models["logistic"].named_steps["model"].random_state == 7
    assert models["hist_gradient_boosting"].named_steps["model"].random_state == 7
    assert models["lightgbm"].named_steps["model"].random_state == 7


def test_pretrain_logs_validation_loss(tmp_path, monkeypatch):
    monkeypatch.setattr("src.models.pretrain.select_device", lambda: torch.device("cpu"))

    def make_tensors(categories):
        categories = torch.tensor(categories)
        shape = categories.shape
        return {
            "category": categories,
            "amount": torch.ones(shape, dtype=torch.float32),
            "relative_amount": torch.ones(shape, dtype=torch.float32),
            "time": torch.zeros(shape, dtype=torch.long),
            "direction": torch.zeros(shape, dtype=torch.long),
            "padding": categories.eq(PAD_ID),
        }

    train_tensors = make_tensors([[0, 3, PAD_ID, PAD_ID], [1, 2, PAD_ID, PAD_ID]])
    validation_tensors = make_tensors([[0, 1, PAD_ID, PAD_ID]])

    _, history = pretrain(
        train_tensors,
        tmp_path / "encoder.pt",
        embedding_dim=8,
        heads=2,
        layers=1,
        epochs=2,
        batch_size=2,
        seed=3,
        validation_tensors=validation_tensors,
    )

    assert len(history) == 2
    assert all(np.isfinite(row["validation_loss"]) for row in history)
    assert (tmp_path / "encoder.pt").exists()
