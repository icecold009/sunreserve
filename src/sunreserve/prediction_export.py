"""Shared portable JSON schema for UI and command-line next-day predictions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_EXPORT_FORMAT = "sunreserve-prediction-export"
PREDICTION_EXPORT_SCHEMA_VERSION = 1
PREDICTION_EXPORT_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "result_status",
        "generated_at_utc",
        "issue_date",
        "target_date",
        "source",
        "observations",
        "assumptions",
        "model_identity",
        "package_version",
        "package_source_identity",
        "entrypoint_identity",
        "forecasts",
        "prediction",
        "reserve_outputs",
        "input_validation_policy",
    }
)
_RESULT_STATUSES = {"fresh", "stale", "not_applicable"}
_APPROACHES = ("ridge", "persistence", "seasonal_mean")
_ASSUMPTION_UNITS = {
    "daily_demand_kwh": "kWh/day",
    "reserve_capacity_kwh": "kWh",
    "reserve_buffer_kwh": "kWh",
    "panel_area_m2": "m²",
    "conversion_efficiency": "fraction",
    "derating_factor": "factor",
}
_FORECAST_FIELDS = (
    "ridge_radiation_kwh_m2_day",
    "persistence_radiation_kwh_m2_day",
    "seasonal_mean_radiation_kwh_m2_day",
)


def generated_at_utc_now() -> str:
    """Return the current UTC time for a prediction run, with an explicit UTC suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def package_source_identity() -> dict[str, str | int]:
    """Hash installed package Python sources without depending on their install path."""
    package_root = Path(__file__).resolve().parent
    sources = sorted(path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts)
    if not sources:
        raise RuntimeError("SunReserve package Python sources are unavailable.")

    digest = hashlib.sha256()
    for source in sources:
        relative_name = source.relative_to(package_root).as_posix().encode("utf-8")
        source_bytes = source.read_bytes()
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    return {
        "identity_type": "installed_package_python_sources",
        "algorithm": "sha256",
        "digest": digest.hexdigest(),
        "file_count": len(sources),
    }


def entrypoint_source_identity(application_kind: str, source_bytes: bytes) -> dict[str, str]:
    """Identify a UI or CLI entrypoint from its source bytes without exposing its path."""
    if application_kind not in {"streamlit_ui", "cli"}:
        raise ValueError("application_kind must be streamlit_ui or cli")
    if not isinstance(source_bytes, bytes):
        raise ValueError("entrypoint source must be supplied as bytes")
    return {
        "application_kind": application_kind,
        "algorithm": "sha256",
        "digest": hashlib.sha256(source_bytes).hexdigest(),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("prediction export contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    raise ValueError(f"prediction export contains an unsupported value: {type(value).__name__}")


def _date_text(value: date | str, field: str) -> str:
    if isinstance(value, datetime):
        raise ValueError(f"{field} must be a calendar date, not a date-time")
    try:
        parsed = date.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError(f"{field} must use ISO YYYY-MM-DD format") from error
    if not isinstance(parsed, date):
        raise ValueError(f"{field} must be a calendar date")
    return parsed.isoformat()


def _timestamp_text(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("generated_at_utc must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("generated_at_utc must include the UTC timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return number


def _observation_records(
    observations: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(observations, pd.DataFrame):
        if "date" in observations.columns:
            raw_records = observations.to_dict(orient="records")
        else:
            if not isinstance(observations.index, pd.DatetimeIndex):
                raise ValueError("observations must include date values")
            raw_records = [
                {"date": timestamp.date().isoformat(), **row}
                for timestamp, row in zip(
                    observations.index, observations.to_dict(orient="records"), strict=True
                )
            ]
    else:
        raw_records = list(observations)
    if len(raw_records) != 7 or any(not isinstance(row, Mapping) for row in raw_records):
        raise ValueError("prediction export requires seven exact observation records")
    return [_json_value(row) for row in raw_records]


def _reserve_records(
    outputs: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(outputs, pd.DataFrame):
        raw_records = outputs.to_dict(orient="records")
    else:
        raw_records = list(outputs)
    if len(raw_records) != len(_APPROACHES) or any(
        not isinstance(row, Mapping) for row in raw_records
    ):
        raise ValueError("prediction export requires reserve outputs for all three approaches")

    normalized = []
    for approach, raw_record in zip(_APPROACHES, raw_records, strict=True):
        row = _json_value(raw_record)
        demand = _finite_number(row["demand_kwh"], "demand_kwh", nonnegative=True)
        solar_proxy = _finite_number(
            row["predicted_generation_proxy_kwh_day"],
            "predicted_generation_proxy_kwh_day",
            nonnegative=True,
        )
        reserve = _finite_number(
            row["prepared_reserve_kwh"], "prepared_reserve_kwh", nonnegative=True
        )
        buffer = _finite_number(row["buffer_kwh"], "buffer_kwh", nonnegative=True)
        row["approach"] = approach
        row["backup_need_before_capacity_cap_kwh"] = max(0.0, demand - solar_proxy + buffer)
        row["predicted_remaining_gap_kwh"] = max(0.0, demand - solar_proxy - reserve)
        normalized.append(row)
    return normalized


def build_prediction_export(
    *,
    generated_at_utc: str,
    issue_date: date | str,
    target_date: date | str,
    source_kind: str,
    observations: pd.DataFrame | Sequence[Mapping[str, Any]],
    assumptions: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    package_version: str,
    package_source: Mapping[str, Any],
    entrypoint_identity: Mapping[str, Any],
    forecasts: Mapping[str, Any],
    reserve_outputs: pd.DataFrame | Sequence[Mapping[str, Any]],
    features: Mapping[str, Any],
    warnings: Sequence[str],
    input_validation_policy: str,
    result_status: str,
) -> dict[str, Any]:
    """Build one strict JSON-ready prediction export for UI and CLI callers."""
    if result_status not in _RESULT_STATUSES:
        raise ValueError(f"unsupported prediction result status {result_status!r}")
    if not isinstance(source_kind, str) or re.fullmatch(r"[a-z][a-z0-9_]*", source_kind) is None:
        raise ValueError("source_kind must be a path-free lowercase identifier")

    if set(assumptions) != set(_ASSUMPTION_UNITS):
        raise ValueError("prediction export assumptions do not match the shared schema")
    normalized_assumptions = {
        key: {
            "value": _finite_number(assumptions[key], key, nonnegative=True),
            "unit": unit,
        }
        for key, unit in _ASSUMPTION_UNITS.items()
    }

    if set(forecasts) != set(_FORECAST_FIELDS):
        raise ValueError("prediction export forecasts do not match the shared schema")
    normalized_forecasts = {
        key: _finite_number(forecasts[key], key, nonnegative=True) for key in _FORECAST_FIELDS
    }

    model_id = model_identity.get("model_id")
    model_sha256 = model_identity.get("artifact_sha256")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("prediction export model identity is missing a model id")
    if not isinstance(model_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", model_sha256) is None:
        raise ValueError("prediction export model identity requires a SHA-256 digest")
    training_rows = model_identity.get("training_rows")
    if type(training_rows) is not int or training_rows <= 0:
        raise ValueError("prediction export model identity requires a positive training row count")
    training_start = _date_text(model_identity["training_start"], "training_start")
    training_end = _date_text(model_identity["training_end"], "training_end")

    normalized_model = {
        "model_id": model_id,
        "artifact_sha256": model_sha256.lower(),
        "training_rows": training_rows,
        "training_period": {
            "start_date": training_start,
            "end_date": training_end,
        },
    }
    if not isinstance(package_version, str) or not package_version.strip():
        raise ValueError("prediction export producer package version is missing")
    if (
        set(package_source) != {"identity_type", "algorithm", "digest", "file_count"}
        or package_source.get("identity_type") != "installed_package_python_sources"
        or package_source.get("algorithm") != "sha256"
        or not isinstance(package_source.get("digest"), str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", package_source["digest"]) is None
        or type(package_source.get("file_count")) is not int
        or package_source["file_count"] <= 0
    ):
        raise ValueError("prediction export producer package identity is invalid")
    normalized_package_source = {
        "identity_type": package_source["identity_type"],
        "algorithm": "sha256",
        "digest": package_source["digest"].lower(),
        "file_count": package_source["file_count"],
    }
    if (
        set(entrypoint_identity) != {"application_kind", "algorithm", "digest"}
        or entrypoint_identity.get("application_kind") not in {"streamlit_ui", "cli"}
        or entrypoint_identity.get("algorithm") != "sha256"
        or not isinstance(entrypoint_identity.get("digest"), str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", entrypoint_identity["digest"]) is None
    ):
        raise ValueError("prediction export entrypoint identity does not match the shared schema")
    normalized_entrypoint = {
        "application_kind": entrypoint_identity["application_kind"],
        "algorithm": "sha256",
        "digest": entrypoint_identity["digest"].lower(),
    }
    source = {
        "kind": source_kind,
        "availability_status": "unverified",
        "availability_note": (
            "Observation release timestamps and availability by the issue date were not verified."
        ),
    }
    result = {
        "format": PREDICTION_EXPORT_FORMAT,
        "schema_version": PREDICTION_EXPORT_SCHEMA_VERSION,
        "result_status": result_status,
        "generated_at_utc": _timestamp_text(generated_at_utc),
        "issue_date": _date_text(issue_date, "issue_date"),
        "target_date": _date_text(target_date, "target_date"),
        "source": source,
        "observations": _observation_records(observations),
        "assumptions": normalized_assumptions,
        "model_identity": normalized_model,
        "package_version": package_version,
        "package_source_identity": normalized_package_source,
        "entrypoint_identity": normalized_entrypoint,
        "forecasts": normalized_forecasts,
        "prediction": {
            "features": _json_value(features),
            "warnings": _json_value(warnings),
        },
        "reserve_outputs": _reserve_records(reserve_outputs),
        "input_validation_policy": input_validation_policy,
    }
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
