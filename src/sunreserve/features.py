"""Leakage-safe issue-date features and chronological target-date splits."""

from __future__ import annotations

from datetime import date
from math import pi
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.observations import RADIATION, validate_observation_frame

TARGET_COLUMN = "target_radiation_kwh_m2_day"
FEATURE_COLUMNS = (
    "radiation_today_kwh_m2_day",
    "radiation_lag_1_kwh_m2_day",
    "radiation_lag_2_kwh_m2_day",
    "radiation_lag_3_kwh_m2_day",
    "radiation_trailing_7d_mean_kwh_m2_day",
    "temperature_today_c",
    "relative_humidity_today_pct",
    "target_month_sin",
    "target_month_cos",
)
AVAILABILITY_NOTE = (
    "Features use historical observation dates no later than issue_date; source release "
    "timestamps were not verified."
)


def _as_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("daily input must use a DatetimeIndex")
    frame = daily.copy()
    if frame.index.has_duplicates:
        raise ValueError("daily input contains duplicate dates")
    if frame.empty:
        raise ValueError("daily input is empty")
    frame.index = frame.index.normalize()
    if frame.index.has_duplicates:
        raise ValueError("daily input contains multiple observations on one calendar date")
    frame = frame.sort_index()
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    return frame.reindex(expected)


def _split_name(target_date: pd.Timestamp, config: ExperimentConfig) -> str:
    if target_date <= pd.Timestamp(config.training_end):
        return "train"
    if config.validation_start <= target_date.date() <= config.validation_end:
        return "validation"
    if target_date.date() >= config.test_start and target_date.date() <= config.end_date:
        return "test"
    raise ValueError(
        f"target date {target_date.date()} is not covered by the configured chronological splits"
    )


def _feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build the shared issue-date feature matrix for batch and single-row inference."""
    radiation = pd.to_numeric(frame[RADIATION], errors="coerce")
    temperature = pd.to_numeric(frame["T2M"], errors="coerce")
    humidity = pd.to_numeric(frame["RH2M"], errors="coerce")
    issue_dates = pd.DatetimeIndex(frame.index)
    target_dates = issue_dates + pd.Timedelta(days=1)
    features = pd.DataFrame(index=issue_dates)
    features["radiation_today_kwh_m2_day"] = radiation
    for lag in (1, 2, 3):
        features[f"radiation_lag_{lag}_kwh_m2_day"] = radiation.shift(lag)
    features["radiation_trailing_7d_mean_kwh_m2_day"] = radiation.rolling(
        window=7, min_periods=7
    ).mean()
    features["temperature_today_c"] = temperature
    features["relative_humidity_today_pct"] = humidity
    month_angle = 2 * pi * (target_dates.month - 1) / 12
    features["target_month_sin"] = np.sin(month_angle)
    features["target_month_cos"] = np.cos(month_angle)
    return features.loc[:, FEATURE_COLUMNS], target_dates


def build_next_day_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Validate a complete seven-day issue-date window and build the next-day row.

    `observations` must have a DatetimeIndex and the source columns `ALLSKY_SFC_SW_DWN`,
    `T2M`, and `RH2M`. The returned row has no label and uses the same shared feature
    builder as the chronological training rows.
    """
    frame = validate_observation_frame(observations)

    features, target_dates = _feature_frame(frame)
    row = features.iloc[[-1]].copy()
    row.insert(0, "issue_date", frame.index[-1])
    row.insert(1, "target_date", target_dates[-1])
    row["feature_window_start"] = frame.index[0]
    row["feature_available_through"] = frame.index[-1]
    row["feature_availability_note"] = AVAILABILITY_NOTE
    return row.reset_index(drop=True)


def build_supervised_rows(
    daily: pd.DataFrame,
    config: ExperimentConfig,
    *,
    target_date_end: date | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row for issue date t and target date t+1.

    Missing dates are first restored as explicit empty calendar rows. A label therefore
    always means the next calendar day, never the next observed row.
    """
    frame = _as_daily_frame(daily)
    required_columns = {RADIATION, "T2M", "RH2M"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"daily input is missing columns: {', '.join(sorted(missing_columns))}")

    radiation = pd.to_numeric(frame[RADIATION], errors="coerce")
    humidity = pd.to_numeric(frame["RH2M"], errors="coerce")
    if (radiation.dropna() < 0).any():
        raise ValueError("daily solar radiation must be nonnegative")
    if ((humidity.dropna() < 0) | (humidity.dropna() > 100)).any():
        raise ValueError("daily relative humidity must be between 0 and 100 percent")

    target = radiation.shift(-1).rename(TARGET_COLUMN)
    issue_dates = pd.DatetimeIndex(frame.index)
    feature_frame, target_dates = _feature_frame(frame)
    target_month = target_dates.month

    result = feature_frame.copy()
    result[TARGET_COLUMN] = target
    result.insert(0, "issue_date", issue_dates)
    result.insert(1, "target_date", target_dates)
    result["feature_window_start"] = issue_dates - pd.Timedelta(days=6)
    result["feature_available_through"] = issue_dates
    result["feature_availability_note"] = AVAILABILITY_NOTE
    result["target_month"] = target_month

    target_end = min(config.end_date, target_date_end or config.end_date)
    if target_end < config.start_date:
        raise ValueError("target_date_end must not precede start_date")
    in_requested_target_range = (result["target_date"].dt.date >= config.start_date) & (
        result["target_date"].dt.date <= target_end
    )
    candidates = result.loc[in_requested_target_range].copy()
    usable = (
        candidates.loc[:, [*FEATURE_COLUMNS, TARGET_COLUMN]]
        .apply(lambda column: np.isfinite(column.astype(float)))
        .all(axis=1)
    )
    excluded: list[dict[str, Any]] = []
    for index, row in candidates.loc[~usable].iterrows():
        missing = [
            column
            for column in (*FEATURE_COLUMNS, TARGET_COLUMN)
            if not np.isfinite(float(row[column]))
        ]
        excluded.append(
            {
                "issue_date": index.date().isoformat(),
                "target_date": row["target_date"].date().isoformat(),
                "missing_or_nonfinite_fields": missing,
            }
        )
    rows = candidates.loc[usable].copy()
    rows["split"] = rows["target_date"].map(lambda value: _split_name(value, config))
    rows = rows.reset_index(drop=True)
    counts = {name: int((rows["split"] == name).sum()) for name in ("train", "validation", "test")}
    report = {
        "candidate_target_rows": int(len(candidates)),
        "eligible_rows": int(len(rows)),
        "excluded_rows": int(len(excluded)),
        "excluded_dates": excluded,
        "eligible_rows_by_split": counts,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "target_alignment": "target_date is exactly issue_date + one calendar day",
        "target_date_end": target_end.isoformat(),
        "feature_availability": AVAILABILITY_NOTE,
        "split_rule": (
            "split by target_date: through training_end=train, validation_start through "
            "validation_end=validation, test_start through end_date=test"
        ),
    }
    return rows, report


def split_supervised_rows(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return named partitions and assert their target-date ranges do not overlap."""
    required = {"split", "target_date"}
    if not required <= set(rows.columns):
        raise ValueError("supervised rows must contain split and target_date columns")
    allowed = {"train", "validation", "test"}
    unknown = set(rows["split"].dropna()) - allowed
    if unknown:
        raise ValueError(f"unknown split labels: {', '.join(sorted(unknown))}")
    partitions = {
        name: rows.loc[rows["split"] == name].copy().reset_index(drop=True)
        for name in ("train", "validation", "test")
    }
    ranges = {
        name: (part["target_date"].min(), part["target_date"].max())
        for name, part in partitions.items()
        if not part.empty
    }
    ordered = [name for name in ("train", "validation", "test") if name in ranges]
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if ranges[earlier][1] >= ranges[later][0]:
            raise ValueError(f"{earlier} and {later} target-date ranges overlap")
    return partitions
