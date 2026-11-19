from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sunreserve.observations import (
    OBSERVATION_POLICY_NOTE,
    RADIATION,
    observation_frame_from_records,
    validate_observation_frame,
)


def records() -> list[dict[str, object]]:
    return [
        {
            "date": stamp.date().isoformat(),
            RADIATION: 5.0 + index / 10,
            "T2M": 22.0 + index / 10,
            "RH2M": 55.0 - index,
        }
        for index, stamp in enumerate(pd.date_range("2020-01-01", periods=7, freq="D"))
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (RADIATION, -999, "NASA POWER -999 fill marker"),
        ("T2M", -999, "NASA POWER -999 fill marker"),
        ("RH2M", -999, "NASA POWER -999 fill marker"),
        (RADIATION, True, "must be numeric"),
        ("T2M", "not numeric", "must be numeric"),
        ("RH2M", float("inf"), "finite values"),
        ("T2M", -273.16, "absolute zero"),
        (RADIATION, -0.1, "must be nonnegative"),
        ("RH2M", 100.1, "between 0 and 100"),
    ],
)
def test_record_contract_rejects_invalid_values(field: str, value: object, message: str) -> None:
    source = records()
    source[2][field] = value
    with pytest.raises(ValueError, match=message):
        observation_frame_from_records(source)


def test_record_contract_rejects_duplicate_gap_invalid_and_future_dates() -> None:
    source = records()
    source[4]["date"] = source[3]["date"]
    with pytest.raises(ValueError, match="duplicate dates"):
        observation_frame_from_records(source)

    source = records()
    source[4]["date"] = "2020-01-06"
    source[5]["date"] = "2020-01-07"
    source[6]["date"] = "2020-01-08"
    with pytest.raises(ValueError, match="consecutive calendar days"):
        observation_frame_from_records(source)

    source = records()
    source[0]["date"] = "2020-02-30"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        observation_frame_from_records(source)

    source = records()
    future_dates = pd.date_range("2030-01-01", periods=7, freq="D")
    for row, stamp in zip(source, future_dates, strict=True):
        row["date"] = stamp.date().isoformat()
    with pytest.raises(ValueError, match="future-dated"):
        observation_frame_from_records(source)


def test_frame_contract_rejects_subdaily_and_duplicate_calendar_dates() -> None:
    source = records()
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "date"} for row in source],
        index=pd.date_range("2020-01-01 01:00", periods=7, freq="D"),
    )
    with pytest.raises(ValueError, match="without a time"):
        validate_observation_frame(frame)

    frame.index = pd.DatetimeIndex(
        [*pd.date_range("2020-01-01", periods=6, freq="D"), pd.Timestamp("2020-01-06")]
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        validate_observation_frame(frame)


def test_valid_unusual_values_are_accepted_and_policy_is_explicit() -> None:
    source = records()
    source[3]["T2M"] = 60.0
    frame = observation_frame_from_records(source, as_of=date(2020, 1, 7))

    assert frame.loc["2020-01-04", "T2M"] == 60.0
    assert "accepted without clipping" in OBSERVATION_POLICY_NOTE
    assert "may extrapolate" in OBSERVATION_POLICY_NOTE
