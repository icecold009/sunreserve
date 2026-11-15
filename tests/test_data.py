from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

from sunreserve import data as data_module
from sunreserve.config import ExperimentConfig
from sunreserve.data import (
    DataError,
    ProjectPaths,
    fetch_raw,
    parse_power_payload,
    prepare_daily,
    sha256_bytes,
)


def config() -> ExperimentConfig:
    return ExperimentConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 3),
        training_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
    )


def document() -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [77.595, 12.972, 841.72]},
        "header": {
            "start": "20240101",
            "end": "20240103",
            "time_standard": "UTC",
        },
        "parameters": {
            "ALLSKY_SFC_SW_DWN": {"units": "kW-hr/m^2/day"},
            "T2M": {"units": "C"},
            "RH2M": {"units": "%"},
        },
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": {"20240101": 4.0, "20240102": -999, "20240103": 2.0},
                "T2M": {"20240101": 22.0, "20240102": 23.0},
                "RH2M": {"20240101": 55.0, "20240102": 60.0, "20240103": 65.0},
            }
        },
    }


def payload(value: dict[str, Any] | None = None) -> bytes:
    return json.dumps(value if value is not None else document()).encode("utf-8")


def test_parse_builds_full_calendar_and_converts_fill_and_absent_values() -> None:
    frame, metadata = parse_power_payload(payload(), config())

    assert list(frame.index) == list(pd.date_range("2024-01-01", "2024-01-03", freq="D"))
    assert frame.loc["2024-01-01", "ALLSKY_SFC_SW_DWN"] == 4.0
    assert np.isnan(frame.loc["2024-01-02", "ALLSKY_SFC_SW_DWN"])
    assert np.isnan(frame.loc["2024-01-03", "T2M"])
    assert metadata["missing_value_marker"] == -999.0
    assert metadata["reported_location"] == {"latitude": 12.972, "longitude": 77.595}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("geometry", "coordinates", 1), 13.0, "latitude"),
        (("geometry", "coordinates", 0), 77.5, "longitude"),
        (("parameters", "T2M", "units"), "F", "unexpected unit"),
        (("header", "time_standard"), "LST", "time standard"),
        (("header", "start"), "20240102", "start date"),
    ],
)
def test_rejects_mismatched_location_units_time_and_dates(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    response = document()
    current = response
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value

    with pytest.raises(DataError, match=message):
        parse_power_payload(payload(response), config())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("properties", "parameter", "ALLSKY_SFC_SW_DWN", "20240102"), -0.1, "negative solar"),
        (("properties", "parameter", "RH2M", "20240102"), 101, "outside 0–100%"),
        (("properties", "parameter", "T2M", "20240101"), "bad", "non-numeric"),
    ],
)
def test_rejects_impossible_or_malformed_observations(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    response = document()
    current = response
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value

    with pytest.raises(DataError, match=message):
        parse_power_payload(payload(response), config())


def test_rejects_wrong_shape_invalid_dates_and_duplicate_json_keys() -> None:
    with pytest.raises(DataError, match="root must be a JSON object"):
        parse_power_payload(b"[]", config())
    with pytest.raises(DataError, match="malformed date key"):
        response = document()
        response["properties"]["parameter"]["T2M"]["2024-01-01"] = 21
        parse_power_payload(payload(response), config())
    with pytest.raises(DataError, match="duplicate JSON key"):
        parse_power_payload(b'{"header": {}, "header": {}}', config())
    duplicate_date = payload().replace(b'"20240103": 2.0', b'"20240101": 2.0', 1)
    with pytest.raises(DataError, match="duplicate JSON key '20240101'"):
        parse_power_payload(duplicate_date, config())
    with pytest.raises(DataError, match="missing parameter RH2M"):
        response = document()
        del response["properties"]["parameter"]["RH2M"]
        parse_power_payload(payload(response), config())


def test_rejects_dates_outside_requested_window() -> None:
    response = document()
    response["properties"]["parameter"]["T2M"]["20231231"] = 20.0

    with pytest.raises(DataError, match="outside requested range"):
        parse_power_payload(payload(response), config())


def test_fetch_preserves_raw_bytes_and_reuses_matching_cache(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    raw = payload()
    called: list[str] = []

    def downloader(url: str) -> bytes:
        called.append(url)
        return raw

    raw_path, digest, reused = fetch_raw(config(), paths, downloader=downloader)
    assert raw_path.read_bytes() == raw
    assert digest == sha256_bytes(raw)
    assert reused is False
    sidecar = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["sha256"] == digest
    assert sidecar["retrieved_utc"].endswith("+00:00")
    assert sidecar["units"]["ALLSKY_SFC_SW_DWN"] == "kW-hr/m^2/day"

    second_path, second_digest, reused_again = fetch_raw(
        config(), paths, downloader=lambda _url: pytest.fail("cache should be reused")
    )
    assert second_path == raw_path
    assert second_digest == digest
    assert reused_again is True
    assert len(called) == 1


def test_corrupt_cache_fails_closed_and_refresh_validates_before_replacing(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    raw = payload()
    fetch_raw(config(), paths, downloader=lambda _url: raw)
    original = paths.raw_path.read_bytes()

    sidecar = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    sidecar["sha256"] = "incorrect"
    paths.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(DataError, match="checksum"):
        fetch_raw(config(), paths, downloader=lambda _url: pytest.fail("invalid cache must fail"))

    with pytest.raises(DataError, match="GeoJSON point"):
        fetch_raw(
            config(),
            paths,
            refresh=True,
            downloader=lambda _url: b'{"header": {}}',
        )
    assert paths.raw_path.read_bytes() == original
    assert json.loads(paths.sidecar_path.read_text(encoding="utf-8"))["sha256"] == "incorrect"


def test_refresh_does_not_replace_valid_cache_when_new_response_is_invalid(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    raw = payload()
    fetch_raw(config(), paths, downloader=lambda _url: raw)
    original = paths.raw_path.read_bytes()
    original_sidecar = paths.sidecar_path.read_bytes()

    with pytest.raises(DataError, match="GeoJSON point"):
        fetch_raw(
            config(),
            paths,
            refresh=True,
            downloader=lambda _url: b'{"header": {}}',
        )
    assert paths.raw_path.read_bytes() == original
    assert paths.sidecar_path.read_bytes() == original_sidecar


def test_prepare_writes_quality_report_with_explicit_missingness_and_exclusions(
    tmp_path: Path,
) -> None:
    paths = ProjectPaths(tmp_path)
    fetch_raw(config(), paths, downloader=lambda _url: payload())

    frame, quality = prepare_daily(config(), paths)
    saved = pd.read_csv(paths.daily_path, parse_dates=["date"]).set_index("date")

    assert len(frame) == 3
    assert saved.index.equals(frame.index)
    assert quality["calendar_days"] == 3
    assert quality["missing_counts_by_variable"] == {
        "ALLSKY_SFC_SW_DWN": 1,
        "T2M": 1,
        "RH2M": 0,
    }
    assert quality["missing_dates_by_variable"] == {
        "ALLSKY_SFC_SW_DWN": ["2024-01-02"],
        "T2M": ["2024-01-03"],
        "RH2M": [],
    }
    assert quality["complete_case_days"] == 1
    assert quality["excluded_days"] == 2
    assert quality["excluded_dates"] == [
        {"date": "2024-01-02", "missing_variables": ["ALLSKY_SFC_SW_DWN"]},
        {"date": "2024-01-03", "missing_variables": ["T2M"]},
    ]
    report = json.loads(paths.quality_path.read_text(encoding="utf-8"))
    assert report["raw_sha256"] == sha256_bytes(paths.raw_path.read_bytes())
    assert report["prepared_daily_csv_sha256"] == sha256_bytes(paths.daily_path.read_bytes())
    assert "do not interpolate" in report["exclusion_policy"]


def test_cache_request_mismatch_and_incomplete_cache_are_actionable(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    paths.ensure_raw()
    paths.raw_path.write_bytes(payload())
    with pytest.raises(DataError, match="only part"):
        fetch_raw(config(), paths)

    paths.sidecar_path.write_text("{}", encoding="utf-8")
    with pytest.raises(DataError, match="does not match"):
        fetch_raw(config(), paths)


def test_download_does_not_retry_invalid_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def failed_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise HTTPError("https://example.invalid", 422, "bad request", {}, None)

    monkeypatch.setattr(data_module, "urlopen", failed_request)
    with pytest.raises(DataError, match="HTTP 422"):
        data_module._download("https://example.invalid")
    assert calls == 1


def test_download_retries_transient_network_failures_with_a_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    pauses: list[float] = []

    def failed_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise URLError("temporary connection failure")

    monkeypatch.setattr(data_module, "urlopen", failed_request)
    monkeypatch.setattr(data_module.time, "sleep", pauses.append)
    with pytest.raises(DataError, match="after 3 attempts"):
        data_module._download("https://example.invalid")
    assert calls == 3
    assert pauses == [0.5, 1.0]
