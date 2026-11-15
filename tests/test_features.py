from datetime import date

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_supervised_rows,
)


def config() -> ExperimentConfig:
    return ExperimentConfig(
        start_date=date(2020, 12, 25),
        end_date=date(2021, 1, 10),
        training_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
    )


def daily_data() -> pd.DataFrame:
    dates = pd.date_range("2020-12-25", "2021-01-10", freq="D")
    return pd.DataFrame(
        {
            "ALLSKY_SFC_SW_DWN": np.arange(1.0, len(dates) + 1),
            "T2M": np.arange(20.0, 20.0 + len(dates)),
            "RH2M": np.arange(50.0, 50.0 + len(dates)),
        },
        index=dates,
    )


def row_for_target(rows: pd.DataFrame, target: str) -> pd.Series:
    return rows.loc[rows["target_date"] == pd.Timestamp(target)].iloc[0]


def test_issue_date_predicts_exact_next_calendar_day_with_trailing_features() -> None:
    rows, report = build_supervised_rows(daily_data(), config())
    row = row_for_target(rows, "2021-01-01")

    assert row["issue_date"] == pd.Timestamp("2020-12-31")
    assert row["target_date"] == pd.Timestamp("2021-01-01")
    assert row["target_radiation_kwh_m2_day"] == 8.0
    assert row["radiation_today_kwh_m2_day"] == 7.0
    assert row["radiation_lag_1_kwh_m2_day"] == 6.0
    assert row["radiation_lag_2_kwh_m2_day"] == 5.0
    assert row["radiation_lag_3_kwh_m2_day"] == 4.0
    assert row["radiation_trailing_7d_mean_kwh_m2_day"] == 4.0
    assert row["temperature_today_c"] == 26.0
    assert row["relative_humidity_today_pct"] == 56.0
    assert row["target_month_sin"] == 0.0
    assert row["target_month_cos"] == 1.0
    assert row["feature_window_start"] == pd.Timestamp("2020-12-25")
    assert row["feature_available_through"] == row["issue_date"]
    assert "release timestamps were not verified" in row["feature_availability_note"]
    assert len(report["feature_columns"]) == len(FEATURE_COLUMNS)
    assert report["target_column"] == TARGET_COLUMN


def test_missing_calendar_day_is_not_skipped_to_find_the_next_observation() -> None:
    daily = daily_data().drop(pd.Timestamp("2020-12-28"))
    rows, report = build_supervised_rows(daily, config())

    # Dec 27's target is the missing Dec 28, and Dec 29's seven-day feature window
    # contains that gap. Neither row may bridge forward to the next observed day.
    excluded_by_target = {item["target_date"]: item for item in report["excluded_dates"]}
    assert "2020-12-28" in excluded_by_target
    assert TARGET_COLUMN in excluded_by_target["2020-12-28"]["missing_or_nonfinite_fields"]
    assert "2020-12-29" in excluded_by_target
    assert rows.loc[rows["target_date"] == pd.Timestamp("2020-12-30")].empty


def test_future_observations_cannot_change_an_earlier_feature_row() -> None:
    original = daily_data()
    changed = original.copy()
    changed.loc[changed.index > pd.Timestamp("2020-12-31"), "ALLSKY_SFC_SW_DWN"] += 500

    original_rows, _ = build_supervised_rows(original, config())
    changed_rows, _ = build_supervised_rows(changed, config())
    earlier_original = row_for_target(original_rows, "2021-01-01")
    earlier_changed = row_for_target(changed_rows, "2021-01-01")

    assert earlier_original.loc[list(FEATURE_COLUMNS)].equals(
        earlier_changed.loc[list(FEATURE_COLUMNS)]
    )
    assert earlier_original[TARGET_COLUMN] != earlier_changed[TARGET_COLUMN]
