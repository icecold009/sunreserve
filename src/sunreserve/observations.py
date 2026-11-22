"""Shared validation for user supplied daily weather observations."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd

RADIATION = "ALLSKY_SFC_SW_DWN"
OBSERVATION_FIELDS = ("date", RADIATION, "T2M", "RH2M")
NUMERIC_FIELDS = (RADIATION, "T2M", "RH2M")
NASA_FILL_VALUE = -999.0
ABSOLUTE_ZERO_C = -273.15

OBSERVATION_POLICY_NOTE = (
    "Input checks enforce known physical limits only; plausible unusual values are accepted "
    "without clipping. The model bundle does not record training-support bounds, so predictions "
    "may extrapolate and should be treated as exploratory."
)


def _row_prefix(row_label: str, position: int) -> str:
    return f"{row_label} {position}"


def _validate_physical_value(value: float, field: str, prefix: str) -> None:
    if value == NASA_FILL_VALUE:
        raise ValueError(f"{prefix} field {field} is the NASA POWER -999 fill marker")
    if field == RADIATION and value < 0:
        raise ValueError(f"{prefix} field {field} must be nonnegative")
    if field == "T2M" and value < ABSOLUTE_ZERO_C:
        raise ValueError(f"{prefix} field T2M is below absolute zero (-273.15 °C)")
    if field == "RH2M" and not 0 <= value <= 100:
        raise ValueError(f"{prefix} field RH2M must be between 0 and 100 percent")


def validate_observation_frame(
    observations: pd.DataFrame,
    *,
    row_label: str = "observation",
    as_of: date | None = None,
) -> pd.DataFrame:
    """Return a normalized numeric frame after the shared seven-day contract checks."""
    if not isinstance(observations, pd.DataFrame):
        raise ValueError("daily observation window must be a pandas DataFrame")
    missing = set(NUMERIC_FIELDS) - set(observations.columns)
    if missing:
        raise ValueError(
            f"daily observation window is missing columns: {', '.join(sorted(missing))}"
        )
    if not isinstance(observations.index, pd.DatetimeIndex):
        raise ValueError("daily observation window must use a DatetimeIndex")
    if len(observations) != 7:
        raise ValueError("daily observation window must contain exactly seven daily rows")
    if observations.index.tz is not None:
        raise ValueError("daily observation dates must be timezone-naive calendar dates")
    if observations.index.hasnans:
        raise ValueError("daily observation window contains an invalid date")
    if observations.index.has_duplicates:
        raise ValueError("daily observation window contains duplicate dates")

    frame = observations.loc[:, list(NUMERIC_FIELDS)]
    index = observations.index
    normalized = index.normalize()
    if not index.equals(normalized):
        raise ValueError("daily observation dates must be calendar dates without a time")
    if not index.is_monotonic_increasing:
        raise ValueError("daily observation dates must be in chronological order")
    expected = pd.date_range(index[0], periods=7, freq="D")
    if not index.equals(expected):
        raise ValueError("daily observation dates must be consecutive calendar days")
    current_date = as_of if as_of is not None else datetime.now(UTC).date()
    if index[-1].date() > current_date:
        raise ValueError("daily observation window must not contain future-dated observations")

    converted: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        prefix = _row_prefix(row_label, row_number)
        for field in NUMERIC_FIELDS:
            raw = row[field]
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{prefix} field {field} must be numeric, not boolean")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{prefix} field {field} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"{prefix} field {field} must contain finite values")
            _validate_physical_value(value, field, prefix)
            converted[field].append(value)

    return pd.DataFrame(converted, index=normalized, dtype="float64")


def observation_frame_from_records(
    records: Any,
    *,
    row_label: str = "observation",
    as_of: date | None = None,
) -> pd.DataFrame:
    """Parse the shared strict record shape used by CSV, UI, and JSON CLI input."""
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError("observations must be a list of exactly seven daily records")

    parsed: list[dict[str, object]] = []
    dates: list[pd.Timestamp] = []
    for position, record in enumerate(records, start=1):
        prefix = _row_prefix(row_label, position)
        if not isinstance(record, dict):
            raise ValueError(f"{prefix} must be a JSON object")
        missing = set(OBSERVATION_FIELDS) - set(record)
        extra = set(record) - set(OBSERVATION_FIELDS)
        if missing:
            raise ValueError(f"{prefix} is missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"{prefix} has unsupported fields: {', '.join(sorted(extra))}")

        raw_date = record["date"]
        if not isinstance(raw_date, str) or len(raw_date) != 10:
            raise ValueError(f"{prefix} date must use ISO YYYY-MM-DD format")
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise ValueError(f"{prefix} date must use ISO YYYY-MM-DD format") from error
        if parsed_date.isoformat() != raw_date:
            raise ValueError(f"{prefix} date must use ISO YYYY-MM-DD format")
        dates.append(pd.Timestamp(parsed_date))
        parsed.append({field: record[field] for field in NUMERIC_FIELDS})

    frame = pd.DataFrame(parsed, index=pd.DatetimeIndex(dates))
    return validate_observation_frame(frame, row_label=row_label, as_of=as_of)
