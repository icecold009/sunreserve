"""NASA POWER daily data download, cache validation, and preparation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from sunreserve.config import ExperimentConfig

API_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
FILL_VALUE = -999.0
SCHEMA_VERSION = 2
EXPECTED_UNITS = {
    "ALLSKY_SFC_SW_DWN": ("kW-hr/m^2/day", "kWh/m^2/day"),
    "T2M": ("C", "°C", "deg C"),
    "RH2M": ("%", "percent"),
}


class DataError(RuntimeError):
    """Raised when source data or cached files fail a required check."""


def default_project_root() -> Path:
    """Return the repository root from this installed-in-tree package."""
    return Path(__file__).resolve().parents[2]


class ProjectPaths:
    """Explicit local paths used by data commands."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_project_root()).resolve()
        stem = f"power_daily_{self.root.name}"
        self.raw_path = self.root / "data" / "raw" / f"{stem}.json"
        self.sidecar_path = self.root / "data" / "raw" / f"{stem}.meta.json"
        self.daily_path = self.root / "data" / "processed" / "daily.csv"
        self.quality_path = self.root / "data" / "processed" / "quality_report.json"
        self.config_path = self.root / "artifacts" / "experiment_config.json"
        self.validation_dir = self.root / "artifacts" / "validation"
        self.runs_dir = self.root / "artifacts" / "runs"
        self.model_path = self.root / "models" / "ridge_model.json"

    def ensure_raw(self) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_processed(self) -> None:
        self.daily_path.parent.mkdir(parents=True, exist_ok=True)


def request_url(config: ExperimentConfig) -> str:
    """Construct the explicit POWER request URL represented by a config."""
    query = urlencode(
        [
            ("parameters", ",".join(config.parameters)),
            ("community", config.community),
            ("longitude", f"{config.longitude:.4f}"),
            ("latitude", f"{config.latitude:.4f}"),
            ("start", config.start_date.strftime("%Y%m%d")),
            ("end", config.end_date.strftime("%Y%m%d")),
            ("format", "JSON"),
            ("time-standard", config.time_standard),
        ]
    )
    return f"{API_ENDPOINT}?{query}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_object(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_json_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataError(f"NASA POWER response is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise DataError("NASA POWER response root must be a JSON object")
    return value


def _normalize_unit(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", value).casefold()


def _parameter_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise DataError("NASA POWER response is missing parameter metadata")
    return parameters


def _header_value(header: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in header:
            return header[name]
    return None


def _assert_header(
    payload: dict[str, Any], config: ExperimentConfig
) -> tuple[dict[str, Any], dict[str, float]]:
    header = payload.get("header")
    if not isinstance(header, dict):
        raise DataError("NASA POWER response is missing a header object")
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        raise DataError("NASA POWER response is missing a GeoJSON point geometry")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        raise DataError("NASA POWER response has no valid point coordinates")
    try:
        longitude = float(coordinates[0])
        latitude = float(coordinates[1])
    except (TypeError, ValueError) as error:
        raise DataError("NASA POWER response has no valid point coordinates") from error
    # POWER rounds the requested point in its geometry to three decimal places.
    if not math.isclose(latitude, config.latitude, abs_tol=0.001):
        raise DataError(
            f"response latitude {latitude} does not match requested latitude {config.latitude}"
        )
    if not math.isclose(longitude, config.longitude, abs_tol=0.001):
        raise DataError(
            f"response longitude {longitude} does not match requested longitude {config.longitude}"
        )
    response_start = _header_value(header, "start")
    response_end = _header_value(header, "end")
    if response_start != config.start_date.strftime("%Y%m%d"):
        raise DataError(f"response start date {response_start!r} does not match request")
    if response_end != config.end_date.strftime("%Y%m%d"):
        raise DataError(f"response end date {response_end!r} does not match request")
    response_time = _header_value(header, "time_standard", "time-standard")
    if response_time is None:
        raise DataError("NASA POWER response is missing its time standard")
    if str(response_time).upper() != config.time_standard:
        raise DataError(
            f"response time standard {response_time!r} does not match {config.time_standard!r}"
        )
    return header, {"latitude": latitude, "longitude": longitude}


def parse_power_payload(
    payload: bytes, config: ExperimentConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a raw POWER JSON response and return an indexed daily frame."""
    document = _json_object(payload)
    header, reported_location = _assert_header(document, config)
    properties = document.get("properties")
    if not isinstance(properties, dict):
        raise DataError("NASA POWER response is missing properties")
    values_by_parameter = properties.get("parameter")
    if not isinstance(values_by_parameter, dict):
        raise DataError("NASA POWER response is missing properties.parameter")
    metadata = _parameter_metadata(document)
    for parameter in config.parameters:
        if parameter not in values_by_parameter:
            raise DataError(f"NASA POWER response is missing parameter {parameter}")
        if parameter not in metadata or not isinstance(metadata[parameter], dict):
            raise DataError(f"NASA POWER response is missing metadata for {parameter}")
        reported_unit = metadata[parameter].get("units")
        allowed_units = EXPECTED_UNITS.get(parameter)
        if allowed_units is None:
            raise DataError(f"no expected unit declaration for parameter {parameter}")
        normalized = _normalize_unit(reported_unit)
        if normalized not in {_normalize_unit(unit) for unit in allowed_units}:
            raise DataError(
                f"unexpected unit for {parameter}: {reported_unit!r}; "
                f"expected one of {allowed_units}"
            )

    expected_dates = pd.date_range(config.start_date, config.end_date, freq="D")
    expected_keys = {item.strftime("%Y%m%d") for item in expected_dates}
    observations: dict[str, dict[pd.Timestamp, float]] = {}
    for parameter in config.parameters:
        values = values_by_parameter[parameter]
        if not isinstance(values, dict):
            raise DataError(f"parameter {parameter} values must be an object keyed by date")
        parsed: dict[pd.Timestamp, float] = {}
        for raw_date, raw_value in values.items():
            if not isinstance(raw_date, str) or not re.fullmatch(r"\d{8}", raw_date):
                raise DataError(f"malformed date key for {parameter}: {raw_date!r}")
            if raw_date not in expected_keys:
                raise DataError(f"date {raw_date} for {parameter} falls outside requested range")
            try:
                stamp = pd.Timestamp(
                    date.fromisoformat(f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}")
                )
            except ValueError as error:
                raise DataError(f"invalid calendar date {raw_date!r}") from error
            if raw_value is None:
                number = math.nan
            elif isinstance(raw_value, bool):
                raise DataError(f"boolean observation is invalid for {parameter} on {raw_date}")
            else:
                try:
                    number = float(raw_value)
                except (TypeError, ValueError) as error:
                    raise DataError(
                        f"non-numeric observation for {parameter} on {raw_date}"
                    ) from error
                if number == FILL_VALUE:
                    number = math.nan
                elif not math.isfinite(number):
                    raise DataError(f"non-finite observation for {parameter} on {raw_date}")
            if math.isfinite(number):
                if parameter == "ALLSKY_SFC_SW_DWN" and number < 0:
                    raise DataError(f"negative solar radiation on {raw_date}")
                if parameter == "RH2M" and not 0 <= number <= 100:
                    raise DataError(f"relative humidity outside 0–100% on {raw_date}")
            parsed[stamp] = number
        observations[parameter] = parsed

    frame = pd.DataFrame(index=expected_dates)
    for parameter, values in observations.items():
        frame[parameter] = pd.Series(values, dtype="float64").reindex(expected_dates)
    frame.index.name = "date"
    return frame, {
        "header": header,
        "reported_location": reported_location,
        "parameters": metadata,
        "missing_value_marker": FILL_VALUE,
    }


def _matching_cache(paths: ProjectPaths, config: ExperimentConfig) -> bytes:
    if not paths.raw_path.exists() or not paths.sidecar_path.exists():
        raise DataError(
            f"cached source data is incomplete at {paths.raw_path}; run fetch or use --refresh"
        )
    raw = paths.raw_path.read_bytes()
    try:
        sidecar = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(
            f"cached provenance is unreadable at {paths.sidecar_path}; use fetch --refresh"
        ) from error
    if sidecar.get("request_url") != request_url(config):
        raise DataError(
            "cached request does not match the active configuration; use fetch --refresh"
        )
    if sidecar.get("sha256") != sha256_bytes(raw):
        raise DataError("cached raw response checksum does not match its provenance sidecar")
    parse_power_payload(raw, config)
    return raw


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _download(url: str, *, attempts: int = 3, timeout_seconds: int = 30) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "SunReserve/0.1 research replay"})
            with urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise DataError(f"NASA POWER returned HTTP {status}")
                return response.read()
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise DataError(f"NASA POWER request failed with HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 == attempts:
                raise DataError(
                    f"NASA POWER request failed after {attempts} attempts: {error}"
                ) from error
        time.sleep(0.5 * (2**attempt))
    raise DataError(f"NASA POWER request failed: {last_error}")


def fetch_raw(
    config: ExperimentConfig,
    paths: ProjectPaths,
    *,
    refresh: bool = False,
    downloader: Callable[[str], bytes] | None = None,
) -> tuple[Path, str, bool]:
    """Fetch or reuse a validated raw response and write its provenance sidecar."""
    if not refresh and paths.raw_path.exists() and paths.sidecar_path.exists():
        raw = _matching_cache(paths, config)
        return paths.raw_path, sha256_bytes(raw), True
    if not refresh and (paths.raw_path.exists() or paths.sidecar_path.exists()):
        raise DataError(
            "only part of the source cache exists; use fetch --refresh to replace it safely"
        )

    url = request_url(config)
    payload = (downloader or _download)(url)
    # Validate fully before touching any existing cache, including a cache being refreshed.
    _, parsed_metadata = parse_power_payload(payload, config)
    digest = sha256_bytes(payload)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "source": config.source,
        "request_url": url,
        "request_parameters": {
            "latitude": config.latitude,
            "longitude": config.longitude,
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "parameters": list(config.parameters),
            "community": config.community,
            "time_standard": config.time_standard,
            "format": "JSON",
        },
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "reported_location": parsed_metadata["reported_location"],
        "units": {name: parsed_metadata["parameters"][name]["units"] for name in config.parameters},
        "missing_value_marker": FILL_VALUE,
        "sha256": digest,
    }
    paths.ensure_raw()
    sidecar_bytes = (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(paths.raw_path, payload)
    _atomic_write(paths.sidecar_path, sidecar_bytes)
    return paths.raw_path, digest, False


def prepare_daily(
    config: ExperimentConfig,
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare a full calendar and missingness report from the validated raw cache."""
    raw = _matching_cache(paths, config)
    frame, _ = parse_power_payload(raw, config)
    frame = frame.sort_index()
    missing_by_variable = {name: int(frame[name].isna().sum()) for name in config.parameters}
    missing_dates_by_variable = {
        name: [stamp.date().isoformat() for stamp in frame.index[frame[name].isna()]]
        for name in config.parameters
    }
    by_year: dict[str, dict[str, int]] = {}
    for year, yearly in frame.groupby(frame.index.year):
        by_year[str(year)] = {name: int(yearly[name].isna().sum()) for name in config.parameters}
    complete = frame.notna().all(axis=1)
    excluded_dates = [
        {
            "date": stamp.date().isoformat(),
            "missing_variables": [
                name for name in config.parameters if pd.isna(frame.loc[stamp, name])
            ],
        }
        for stamp in frame.index[~complete]
    ]
    quality = {
        "schema_version": SCHEMA_VERSION,
        "source": config.source,
        "requested_location": {"latitude": config.latitude, "longitude": config.longitude},
        "time_standard": config.time_standard,
        "requested_start": config.start_date.isoformat(),
        "requested_end": config.end_date.isoformat(),
        "calendar_days": len(frame),
        "units": {name: EXPECTED_UNITS[name][0] for name in config.parameters},
        "missing_value_marker": FILL_VALUE,
        "missing_counts_by_variable": missing_by_variable,
        "missing_dates_by_variable": missing_dates_by_variable,
        "missing_counts_by_year": by_year,
        "complete_case_days": int(complete.sum()),
        "excluded_days": int((~complete).sum()),
        "excluded_dates": excluded_dates,
        "exclusion_policy": (
            "retain the complete calendar and mark source fill values or absent dates as missing; "
            "exclude rows with any missing required value from supervised examples; do not "
            "interpolate or use future observations to fill gaps"
        ),
        "raw_sha256": sha256_bytes(raw),
    }
    daily_content = frame.rename_axis("date").to_csv(
        index_label="date", date_format="%Y-%m-%d", lineterminator="\n"
    )
    quality["prepared_daily_csv_sha256"] = sha256_bytes(daily_content.encode("utf-8"))
    paths.ensure_processed()
    _atomic_write(paths.daily_path, daily_content.encode("utf-8"))
    _atomic_write(
        paths.quality_path,
        (json.dumps(quality, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return frame, quality
