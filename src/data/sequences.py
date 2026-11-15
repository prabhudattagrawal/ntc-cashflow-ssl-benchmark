"""Encode transactions into fixed-width arrays for the transformer.

The IDs are fixed here, nothing is fitted on the full dataset. Sequences come
back as NumPy arrays instead of lists because a few hundred thousand applicants
of list-of-lists is several gigabytes of pure overhead.
"""
import numpy as np
import pandas as pd
import torch

CATEGORIES = [
    "SALARY",
    "BUSINESS_INCOME",
    "RENT",
    "GROCERIES",
    "UTILITIES",
    "TRAVEL",
    "ONLINE_SHOPPING",
    "MEDICAL",
    "TRANSFER",
    "OTHER",
]
CAT_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}
MASK_ID = len(CATEGORIES)
PAD_ID = len(CATEGORIES) + 1
N_CATEGORIES = len(CATEGORIES) + 2
N_TIME_BUCKETS = 16


def encode_sequences(applicants, transactions, max_length=128):
    """Encode each applicant's most recent ``max_length`` transactions.

    Returns ``{applicant_id: (category, amount, relative_amount, time,
    direction, padding)}``, all six entries being arrays of length max_length.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least one")
    if len(transactions) == 0:
        return {}

    app = applicants[["applicant_id", "application_date"]]
    application_dates = pd.Series(
        pd.to_datetime(app["application_date"]).to_numpy(),
        index=pd.Index(app["applicant_id"].to_numpy()),
    )

    applicant_id = transactions["applicant_id"].to_numpy()
    tx_dates = pd.to_datetime(transactions["transaction_date"]).to_numpy()

    # Map the application date onto each transaction instead of merging, which
    # at this scale would copy the whole transaction frame.
    application_date = application_dates.reindex(applicant_id).to_numpy()
    keep = (~pd.isna(application_date)) & (tx_dates < application_date)
    if not keep.any():
        return {}

    amounts = transactions["amount"].to_numpy()
    categories = transactions["category"].to_numpy()
    is_credit = transactions["is_credit"].to_numpy()

    dates = pd.DatetimeIndex(tx_dates)
    tx_day = dates.day.to_numpy()
    tx_month = dates.month.to_numpy()

    # Sort by applicant then date so the tail of each block is the recent history.
    order = np.lexsort((tx_dates, applicant_id))
    order = order[keep[order]]

    ids_sorted = applicant_id[order]
    n_rows = ids_sorted.size
    if n_rows == 0:
        return {}

    new_applicant = np.empty(n_rows, dtype=bool)
    new_applicant[0] = True
    np.not_equal(ids_sorted[1:], ids_sorted[:-1], out=new_applicant[1:])
    starts = np.flatnonzero(new_applicant)
    ends = np.append(starts[1:], n_rows)

    fallback_category = CAT_TO_ID["OTHER"]
    sequences = {}

    for start, end in zip(starts, ends):
        positions = order[start:end][-max_length:]
        length = positions.size

        category_row = np.full(max_length, PAD_ID, dtype=np.int64)
        amount_row = np.zeros(max_length, dtype=np.float32)
        relative_row = np.zeros(max_length, dtype=np.float32)
        time_row = np.zeros(max_length, dtype=np.int64)
        direction_row = np.zeros(max_length, dtype=np.int64)
        padding_row = np.ones(max_length, dtype=bool)

        category_row[:length] = np.fromiter(
            (CAT_TO_ID.get(value, fallback_category) for value in categories[positions]),
            dtype=np.int64,
            count=length,
        )

        amount = np.abs(amounts[positions].astype(np.float64, copy=False))
        positive = amount[amount > 0]
        # Normalising by this applicant's median keeps relative_amount about
        # transaction size, not about income level.
        scale = float(np.median(positive)) if positive.size else 1.0

        day = tx_day[positions]
        month = tx_month[positions]
        time_id = np.minimum((day - 1) // 7, 3) + 4 * ((month - 1) % 4)

        amount_row[:length] = np.log1p(amount)
        relative_row[:length] = np.log1p(amount / scale)
        time_row[:length] = time_id
        direction_row[:length] = is_credit[positions]
        padding_row[:length] = False

        sequences[int(ids_sorted[start])] = (
            category_row,
            amount_row,
            relative_row,
            time_row,
            direction_row,
            padding_row,
        )

    return sequences


def to_tensors(sequence_dict, applicant_ids, max_length=128):
    if not len(applicant_ids):
        return {
            "category": torch.empty((0, max_length), dtype=torch.long),
            "amount": torch.empty((0, max_length), dtype=torch.float32),
            "relative_amount": torch.empty((0, max_length), dtype=torch.float32),
            "time": torch.empty((0, max_length), dtype=torch.long),
            "direction": torch.empty((0, max_length), dtype=torch.long),
            "padding": torch.empty((0, max_length), dtype=torch.bool),
        }

    rows = [sequence_dict[int(a)] for a in applicant_ids]
    cats, amts, relative_amts, times, dirs, pads = zip(*rows)

    def stack(values, dtype):
        array = np.stack(values)
        if array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        return torch.from_numpy(np.ascontiguousarray(array))

    return {
        "category": stack(cats, np.int64),
        "amount": stack(amts, np.float32),
        "relative_amount": stack(relative_amts, np.float32),
        "time": stack(times, np.int64),
        "direction": stack(dirs, np.int64),
        "padding": stack(pads, bool),
    }
