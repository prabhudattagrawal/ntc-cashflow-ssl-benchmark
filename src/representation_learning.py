"""Self-supervised stage: pretrain the transaction encoder and embed applicants."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.sequences import encode_sequences, to_tensors
from src.models.pretrain import extract_embeddings, pretrain


def pretrain_encoder(applicants, transactions, train_ids, cfg, seed, out_dir, epochs=None):
    """Pretrain on training-vintage sequences only.

    Returns the fitted encoder and an audit dict with the id groups used, so
    the caller can check that no held-out applicant was seen.
    """
    model_cfg = cfg["model"]
    max_length = int(model_cfg["max_sequence_length"])

    sequences = encode_sequences(
        applicants[applicants["applicant_id"].isin(train_ids)],
        transactions,
        max_length,
    )

    fit_ids, validation_ids = train_test_split(
        train_ids,
        test_size=float(model_cfg["pretraining_validation_fraction"]),
        random_state=seed,
    )

    model, history = pretrain(
        to_tensors(sequences, fit_ids, max_length),
        out_dir / "transaction_encoder.pt",
        embedding_dim=int(model_cfg["embedding_dim"]),
        heads=int(model_cfg["attention_heads"]),
        layers=int(model_cfg["transformer_layers"]),
        epochs=epochs or int(model_cfg["epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        learning_rate=float(model_cfg["learning_rate"]),
        seed=seed,
        validation_tensors=to_tensors(sequences, validation_ids, max_length),
        early_stopping_patience=int(model_cfg["early_stopping_patience"]),
        max_length=max_length,
    )

    pd.DataFrame(history).to_csv(out_dir / "pretraining_history.csv", index=False)

    audit = {
        "pretrain_ids": set(sequences),
        "fit_ids": fit_ids,
        "validation_ids": validation_ids,
    }
    return model, audit


def embed_applicants(model, applicants, transactions, applicant_ids, max_length):
    """Extract one embedding row per applicant. Returns (frame, column names)."""
    sequences = encode_sequences(applicants, transactions, int(max_length))
    embeddings = extract_embeddings(
        model, to_tensors(sequences, applicant_ids, int(max_length))
    )

    columns = [f"finrep_{i:03d}" for i in range(embeddings.shape[1])]
    frame = pd.DataFrame(embeddings, columns=columns)
    frame["applicant_id"] = applicant_ids
    return frame, columns
