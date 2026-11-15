from datetime import date

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.features import build_supervised_rows, split_supervised_rows


def test_splits_are_chronological_by_target_date_and_use_same_eligible_rows() -> None:
    start = pd.Timestamp("2015-12-25")
    end = pd.Timestamp("2024-12-31")
    dates = pd.date_range(start, end, freq="D")
    daily = pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": np.arange(1.0, len(dates) + 1),
            "T2M": 20.0,
            "RH2M": 50.0,
        },
        index=dates,
    )
    config = ExperimentConfig(
        start_date=date(2016, 1, 1),
        end_date=date(2024, 12, 31),
    )

    rows, report = build_supervised_rows(daily, config)
    partitions = split_supervised_rows(rows)

    assert report["eligible_rows_by_split"] == {
        "train": 2192,
        "validation": 365,
        "test": 731,
    }
    assert len(rows) == 3288
    assert partitions["train"]["target_date"].max() == pd.Timestamp("2021-12-31")
    assert partitions["validation"]["target_date"].min() == pd.Timestamp("2022-01-01")
    assert partitions["validation"]["target_date"].max() == pd.Timestamp("2022-12-31")
    assert partitions["test"]["target_date"].min() == pd.Timestamp("2023-01-01")
    assert partitions["test"]["target_date"].max() == pd.Timestamp("2024-12-31")
    assert partitions["train"]["target_date"].max() < partitions["validation"]["target_date"].min()
    assert partitions["validation"]["target_date"].max() < partitions["test"]["target_date"].min()


def test_target_date_crossing_year_boundary_controls_partition() -> None:
    dates = pd.date_range("2021-12-20", "2022-01-10", freq="D")
    daily = pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": np.arange(1.0, len(dates) + 1),
            "T2M": 20.0,
            "RH2M": 50.0,
        },
        index=dates,
    )
    config = ExperimentConfig(
        start_date=date(2021, 12, 25),
        end_date=date(2022, 1, 10),
    )

    rows, _ = build_supervised_rows(daily, config)
    train = rows.loc[rows["target_date"] == pd.Timestamp("2021-12-31")].iloc[0]
    validation = rows.loc[rows["target_date"] == pd.Timestamp("2022-01-01")].iloc[0]

    assert train["issue_date"] == pd.Timestamp("2021-12-30")
    assert train["split"] == "train"
    assert validation["issue_date"] == pd.Timestamp("2021-12-31")
    assert validation["split"] == "validation"
