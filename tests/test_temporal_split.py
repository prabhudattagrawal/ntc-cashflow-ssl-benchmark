import pandas as pd

from src.data.dataset import temporal_split, temporal_split_with_calibration


def test_split_keeps_dates_together():
    df = pd.DataFrame({
        "applicant_id": range(12),
        "application_date": pd.to_datetime(
            ["2024-01-01"] * 4 + ["2024-02-01"] * 4 + ["2024-03-01"] * 4
        ),
        "default_12m": [0, 1] * 6,
    })

    train, valid, test = temporal_split(df, .33, .33)

    assert train.application_date.max() < valid.application_date.min()
    assert valid.application_date.max() < test.application_date.min()
    assert set(train.applicant_id).isdisjoint(valid.applicant_id)
    assert set(train.applicant_id).isdisjoint(test.applicant_id)


def test_calibration_split_adds_a_later_vintage():
    dates = pd.to_datetime(
        [f"2024-{month:02d}-01" for month in range(1, 11) for _ in range(2)]
    )
    df = pd.DataFrame({"applicant_id": range(20), "application_date": dates})

    train, valid, calibration, test = temporal_split_with_calibration(
        df, valid_fraction=.2, calibration_fraction=.2, test_fraction=.2
    )

    assert train.application_date.max() < valid.application_date.min()
    assert valid.application_date.max() < calibration.application_date.min()
    assert calibration.application_date.max() < test.application_date.min()
    assert set(train.applicant_id).isdisjoint(valid.applicant_id)
    assert set(valid.applicant_id).isdisjoint(calibration.applicant_id)
    assert set(calibration.applicant_id).isdisjoint(test.applicant_id)
