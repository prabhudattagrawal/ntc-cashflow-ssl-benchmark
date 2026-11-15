import numpy as np
import pandas as pd

from src.data.sequences import PAD_ID, encode_sequences, to_tensors


def test_sequence_is_chronological_and_padded():
    app = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    tx = pd.DataFrame({
        "applicant_id": [1, 1],
        "transaction_date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-02")],
        "amount": [100, -50],
        "category": ["SALARY", "GROCERIES"],
        "is_credit": [1, 0],
        "balance_after": [100, 50],
    })

    seq = encode_sequences(app, tx, max_length=4)

    assert seq[1][0][:2].tolist() == [0, 3]
    assert seq[1][0][2:].tolist() == [PAD_ID, PAD_ID]
    assert seq[1][1][0] > seq[1][1][1]

    tensors = to_tensors(seq, [1], max_length=4)
    assert tensors["category"].shape == (1, 4)
    assert tensors["padding"].sum().item() == 2
    assert tensors["amount"].dtype.is_floating_point
    assert tensors["relative_amount"].shape == (1, 4)


def test_truncation_keeps_latest_transactions():
    app = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    # Rows are given out of order on purpose.
    tx = pd.DataFrame({
        "applicant_id": [1, 1, 1],
        "transaction_date": [
            pd.Timestamp("2025-02-03"),
            pd.Timestamp("2025-02-01"),
            pd.Timestamp("2025-02-02"),
        ],
        "amount": [-30, 100, -20],
        "category": ["OTHER", "SALARY", "GROCERIES"],
        "is_credit": [0, 1, 0],
        "balance_after": [50, 100, 80],
    })

    sequence = encode_sequences(app, tx, max_length=2)[1]

    assert sequence[0].tolist() == [3, 9]
    assert sequence[5].tolist() == [False, False]


def test_amounts_are_relative():
    app = pd.DataFrame({
        "applicant_id": [1],
        "application_date": [pd.Timestamp("2025-03-01")],
    })
    tx = pd.DataFrame({
        "applicant_id": [1, 1],
        "transaction_date": [pd.Timestamp("2025-02-01"), pd.Timestamp("2025-02-02")],
        "amount": [1_000_000.0, 2_000_000.0],
        "category": ["SALARY", "SALARY"],
        "is_credit": [1, 1],
        "balance_after": [1_000_000.0, 3_000_000.0],
    })

    sequence = encode_sequences(app, tx, max_length=2)[1]

    assert sequence[1][1] > sequence[1][0]
    amount_scale = 1_500_000.0
    assert np.allclose(sequence[2], [
        np.log1p(1_000_000.0 / amount_scale),
        np.log1p(2_000_000.0 / amount_scale),
    ])
