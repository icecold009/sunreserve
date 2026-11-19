"""Validation-only model comparison, metrics, and reproducible selection artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import shutil
import tempfile
from copy import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import (
    EXPECTED_UNITS,
    SCHEMA_VERSION,
    DataError,
    ProjectPaths,
    _normalize_unit,
    request_url,
)
from sunreserve.decision import (
    BUFFER_FRACTIONS,
    CAPACITY_LEVELS_KWH,
    DEMAND_LEVELS_KWH,
    RADIATION_COLUMN,
    evaluate_decision_scenarios,
)
from sunreserve.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    _split_name,
    build_supervised_rows,
    split_supervised_rows,
)
from sunreserve.inference import (
    MODEL_ID,
    ModelArtifactError,
    RidgeModelBundle,
    load_model_bundle,
    predict_ridge_bundle,
    predict_seasonal_mean_bundle,
    write_model_bundle,
)
from sunreserve.source_provenance import source_provenance
from sunreserve.train import baseline_predictions, fit_ridge, predict_persistence, predict_ridge

ARTIFACT_SCHEMA_VERSION = 2
FINAL_RUN_SCHEMA_VERSION = 1
FINAL_FORECAST_APPROACHES = {"persistence", "seasonal_mean", "ridge"}
FINAL_DECISION_APPROACHES = {*FINAL_FORECAST_APPROACHES, "always_prepare_backup"}
FINAL_PREDICTION_COLUMNS = {
    "artifact_schema_version",
    "issue_date",
    "target_date",
    "split",
    "actual_radiation_kwh_m2_day",
    "persistence_prediction_kwh_m2_day",
    "seasonal_mean_prediction_kwh_m2_day",
    "ridge_prediction_kwh_m2_day",
    "selected_model",
}
FINAL_FORECAST_METRIC_FIELDS = {
    "sample_count",
    "mae",
    "rmse",
    "bias_prediction_minus_actual",
    "monthly_mae_by_target_calendar_month",
    "monthly_sample_count",
    "seasonal_mae",
    "seasonal_sample_count",
    "low_solar",
    "prediction_clip_rule",
}
FINAL_DECISION_SUMMARY_FIELDS = {
    "scenario_id",
    "approach",
    "selected_model",
    "sample_count",
    "simulated_shortfall_day_count",
    "simulated_shortfall_day_share",
    "total_simulated_shortfall_kwh",
    "mean_simulated_shortfall_kwh_day",
    "total_excess_prepared_reserve_kwh",
    "mean_excess_prepared_reserve_kwh_day",
    "capacity_limited_shortfall_day_count",
    "total_capacity_limited_shortfall_kwh",
    "mean_capacity_limited_shortfall_kwh_day",
    "total_additional_shortfall_above_capacity_floor_kwh",
    "mean_additional_shortfall_above_capacity_floor_kwh_day",
    "paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day",
    "paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day",
    "paired_mean_excess_difference_vs_persistence_kwh_day",
    "paired_mean_excess_difference_vs_seasonal_mean_kwh_day",
}
FINAL_DECISION_METRIC_COLUMNS = {
    "artifact_schema_version",
    "partition",
    *FINAL_DECISION_SUMMARY_FIELDS,
    "demand_kwh",
    "capacity_kwh",
    "buffer_kwh",
}
FINAL_DECISION_PREDICTION_COLUMNS = {
    "artifact_schema_version",
    "split",
    "scenario_id",
    "approach",
    "target_date",
    "actual_radiation_kwh_m2_day",
    "predicted_radiation_kwh_m2_day",
    "actual_generation_proxy_kwh_day",
    "predicted_generation_proxy_kwh_day",
    "demand_kwh",
    "capacity_kwh",
    "buffer_kwh",
    "prepared_reserve_kwh",
    "simulated_shortfall_kwh",
    "excess_prepared_reserve_kwh",
    "capacity_limited_shortfall_kwh",
    "additional_shortfall_above_capacity_floor_kwh",
}
SELECTION_TIE_ORDER = ("persistence", "seasonal_mean", "ridge")
DECISION_SELECTION_TIE_ORDER = SELECTION_TIE_ORDER
DECISION_SELECTION_OBJECTIVE = (
    "mean simulated shortfall kWh/day + mean excess prepared reserve kWh/day; "
    "equal 1:1 energy weights, not monetary costs"
)
PREDICTION_CLIP_RULE = "max(0, prediction), applied to every candidate before scoring"
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


@dataclass(frozen=True)
class ValidationSelection:
    selected_model: str
    low_solar_threshold: float
    metrics: dict[str, dict[str, Any]]
    predictions: pd.DataFrame
    eligible_rows: dict[str, int]
    excluded_dates: list[dict[str, Any]]


def _finite_vector(values: Any, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def clip_predictions(predictions: Any) -> np.ndarray:
    """Apply the documented nonnegative radiation forecast rule."""
    return np.maximum(_finite_vector(predictions, "predictions"), 0.0)


def _mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - actual)))


def _rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - actual))))


def forecast_metrics(
    actual: Any,
    predictions: Any,
    target_dates: Any,
    *,
    low_solar_threshold: float,
) -> dict[str, Any]:
    """Compute errors on one common partition with bias defined as prediction − actual."""
    observed = _finite_vector(actual, "actual")
    forecast = clip_predictions(predictions)
    dates = pd.DatetimeIndex(pd.to_datetime(target_dates))
    if len(observed) == 0:
        raise ValueError("cannot calculate metrics for an empty partition")
    if not (len(observed) == len(forecast) == len(dates)):
        raise ValueError("actuals, predictions, and target dates must have equal lengths")
    if dates.has_duplicates:
        raise ValueError("metric target dates must be unique")
    if (observed < 0).any():
        raise ValueError("actual solar radiation must be nonnegative")
    if not np.isfinite(low_solar_threshold) or low_solar_threshold < 0:
        raise ValueError("low_solar_threshold must be finite and nonnegative")

    error = forecast - observed
    absolute_error = np.abs(error)
    frame = pd.DataFrame(
        {
            "target_date": dates,
            "actual": observed,
            "prediction": forecast,
            "absolute_error": absolute_error,
            "error": error,
        }
    )
    month_mae = frame.groupby(frame["target_date"].dt.month)["absolute_error"].mean()
    month_counts = frame.groupby(frame["target_date"].dt.month).size()
    seasonal_mae: dict[str, float] = {}
    seasonal_counts: dict[str, int] = {}
    for season, months in SEASONS.items():
        selected = frame.loc[frame["target_date"].dt.month.isin(months)]
        if not selected.empty:
            seasonal_mae[season] = _mae(
                selected["actual"].to_numpy(), selected["prediction"].to_numpy()
            )
            seasonal_counts[season] = int(len(selected))

    low_solar = frame.loc[frame["actual"] <= low_solar_threshold]
    low_actual = low_solar["actual"].to_numpy()
    low_prediction = low_solar["prediction"].to_numpy()
    low_error = low_prediction - low_actual
    return {
        "sample_count": int(len(observed)),
        "mae": _mae(observed, forecast),
        "rmse": _rmse(observed, forecast),
        "bias_prediction_minus_actual": float(np.mean(error)),
        "monthly_mae_by_target_calendar_month": {
            f"{int(month):02d}": float(value) for month, value in month_mae.items()
        },
        "monthly_sample_count": {
            f"{int(month):02d}": int(value) for month, value in month_counts.items()
        },
        "seasonal_mae": seasonal_mae,
        "seasonal_sample_count": seasonal_counts,
        "low_solar": {
            "threshold_kwh_m2_day": float(low_solar_threshold),
            "sample_count": int(len(low_solar)),
            "mae": _mae(low_actual, low_prediction) if len(low_solar) else None,
            "rmse": _rmse(low_actual, low_prediction) if len(low_solar) else None,
            "bias_prediction_minus_actual": float(np.mean(low_error)) if len(low_solar) else None,
        },
        "prediction_clip_rule": PREDICTION_CLIP_RULE,
    }


def select_validation_model(
    training_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    config: ExperimentConfig,
) -> ValidationSelection:
    """Fit on train only and select among all three approaches using validation MAE."""
    if training_rows.empty or set(training_rows.get("split", pd.Series(dtype=str))) != {"train"}:
        raise ValueError("training partition must contain training rows only")
    if validation_rows.empty or set(validation_rows.get("split", pd.Series(dtype=str))) != {
        "validation"
    }:
        raise ValueError("selection partition must contain validation rows only")
    training_targets = _finite_vector(training_rows[TARGET_COLUMN], "training targets")
    if (training_targets < 0).any():
        raise ValueError("training solar-radiation targets must be nonnegative")
    validation_dates = pd.to_datetime(validation_rows["target_date"])
    if validation_dates.duplicated().any():
        raise ValueError("validation target dates must be unique")
    if (validation_dates.dt.date < config.validation_start).any() or (
        validation_dates.dt.date > config.validation_end
    ).any():
        raise ValueError("validation rows fall outside configured validation dates")

    low_solar_threshold = float(
        np.quantile(training_targets, config.low_solar_quantile, method="linear")
    )
    predictions = baseline_predictions(training_rows, validation_rows)
    ridge = fit_ridge(training_rows, config)
    predictions["ridge_prediction_kwh_m2_day"] = predict_ridge(ridge, validation_rows)

    actual = _finite_vector(validation_rows[TARGET_COLUMN], "validation targets")
    prediction_columns = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }
    metrics = {
        model_name: forecast_metrics(
            actual,
            predictions[column],
            predictions["target_date"],
            low_solar_threshold=low_solar_threshold,
        )
        for model_name, column in prediction_columns.items()
    }
    selected = min(
        SELECTION_TIE_ORDER,
        key=lambda name: (metrics[name]["mae"], SELECTION_TIE_ORDER.index(name)),
    )
    if not predictions["target_date"].equals(validation_rows["target_date"]):
        raise RuntimeError("candidate predictions do not use identical validation dates")
    return ValidationSelection(
        selected_model=selected,
        low_solar_threshold=low_solar_threshold,
        metrics=metrics,
        predictions=predictions,
        eligible_rows={
            "train": int(len(training_rows)),
            "validation": int(len(validation_rows)),
        },
        excluded_dates=[],
    )


def _read_provenance_object(path: Path, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DataError(f"{label} is unreadable or invalid: {error}") from error
    if not isinstance(value, dict):
        raise DataError(f"{label} must contain a JSON object")
    return value


def _require_provenance_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = fields - set(value)
    if missing:
        raise DataError(f"{label} is missing required fields: {', '.join(sorted(missing))}")


def _validate_provenance_location(
    value: Any,
    *,
    label: str,
    config: ExperimentConfig,
    tolerance: float | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != {"latitude", "longitude"}:
        raise DataError(f"{label} must contain latitude and longitude")
    expected = {"latitude": config.latitude, "longitude": config.longitude}
    for coordinate, expected_value in expected.items():
        actual = value[coordinate]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise DataError(f"{label} {coordinate} must be numeric")
        if not math.isfinite(float(actual)):
            raise DataError(f"{label} {coordinate} must be finite")
        matches = (
            math.isclose(float(actual), expected_value, abs_tol=tolerance)
            if tolerance is not None
            else actual == expected_value
        )
        if not matches:
            raise DataError(f"{label} does not match active configuration")


def _validate_provenance_units(value: Any, *, label: str, config: ExperimentConfig) -> None:
    if not isinstance(value, dict) or set(value) != set(config.parameters):
        raise DataError(f"{label} must declare units for every configured parameter")
    for parameter in config.parameters:
        allowed = EXPECTED_UNITS.get(parameter)
        if allowed is None or _normalize_unit(value[parameter]) not in {
            _normalize_unit(unit) for unit in allowed
        }:
            raise DataError(f"{label} has incompatible units for {parameter}")


def _validate_development_provenance(
    paths: ProjectPaths, config: ExperimentConfig
) -> tuple[dict[str, Any], bytes]:
    quality = _read_provenance_object(paths.quality_path, "quality report")
    sidecar = _read_provenance_object(paths.sidecar_path, "raw provenance sidecar")
    quality_fields = {
        "schema_version",
        "source",
        "requested_location",
        "time_standard",
        "requested_start",
        "requested_end",
        "calendar_days",
        "units",
        "missing_value_marker",
        "missing_counts_by_variable",
        "missing_dates_by_variable",
        "missing_counts_by_year",
        "complete_case_days",
        "excluded_days",
        "excluded_dates",
        "exclusion_policy",
        "raw_sha256",
        "prepared_daily_csv_sha256",
    }
    sidecar_fields = {
        "schema_version",
        "source",
        "request_url",
        "request_parameters",
        "retrieved_utc",
        "reported_location",
        "units",
        "missing_value_marker",
        "sha256",
    }
    _require_provenance_fields(quality, quality_fields, "quality report")
    _require_provenance_fields(sidecar, sidecar_fields, "raw provenance sidecar")

    for value, label in ((quality, "quality report"), (sidecar, "raw provenance sidecar")):
        if type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise DataError(f"{label} schema version is unsupported")
        if value["source"] != config.source:
            raise DataError(f"{label} source does not match active configuration")
        if value["missing_value_marker"] != -999.0:
            raise DataError(f"{label} missing-value marker is incompatible")

    if quality["requested_start"] != config.start_date.isoformat():
        raise DataError("prepared data start date does not match active configuration")
    if quality["requested_end"] != config.end_date.isoformat():
        raise DataError("prepared data end date does not match active configuration")
    if quality["time_standard"] != config.time_standard:
        raise DataError("prepared data time standard does not match active configuration")
    _validate_provenance_location(
        quality["requested_location"], label="quality report requested location", config=config
    )
    _validate_provenance_location(
        sidecar["reported_location"],
        label="raw sidecar reported location",
        config=config,
        tolerance=0.001,
    )
    _validate_provenance_units(quality["units"], label="quality report", config=config)
    _validate_provenance_units(sidecar["units"], label="raw sidecar", config=config)

    expected_days = (config.end_date - config.start_date).days + 1
    if type(quality["calendar_days"]) is not int or quality["calendar_days"] != expected_days:
        raise DataError("quality report calendar_days does not match the configured date range")
    if not isinstance(sidecar["retrieved_utc"], str) or not sidecar["retrieved_utc"]:
        raise DataError("raw sidecar retrieved_utc must be a nonempty string")

    expected_parameters = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "parameters": list(config.parameters),
        "community": config.community,
        "time_standard": config.time_standard,
        "format": "JSON",
    }
    if sidecar["request_parameters"] != expected_parameters:
        raise DataError("raw sidecar request parameters do not match active configuration")
    if sidecar["request_url"] != request_url(config):
        raise DataError("raw data provenance URL does not match active configuration")

    parameters = set(config.parameters)
    missing_counts = quality["missing_counts_by_variable"]
    missing_dates = quality["missing_dates_by_variable"]
    missing_by_year = quality["missing_counts_by_year"]
    if not isinstance(missing_counts, dict) or set(missing_counts) != parameters:
        raise DataError("quality report missing_counts_by_variable has an invalid schema")
    if not isinstance(missing_dates, dict) or set(missing_dates) != parameters:
        raise DataError("quality report missing_dates_by_variable has an invalid schema")
    for parameter in config.parameters:
        count = missing_counts[parameter]
        dates = missing_dates[parameter]
        if type(count) is not int or count < 0 or not isinstance(dates, list):
            raise DataError(f"quality report missingness for {parameter} has an invalid schema")
        if count != len(dates) or not all(isinstance(item, str) for item in dates):
            raise DataError(f"quality report missing dates for {parameter} are inconsistent")
    if not isinstance(missing_by_year, dict) or not all(
        isinstance(year, str)
        and isinstance(counts, dict)
        and set(counts) == parameters
        and all(type(count) is int and count >= 0 for count in counts.values())
        for year, counts in missing_by_year.items()
    ):
        raise DataError("quality report missing_counts_by_year has an invalid schema")
    for field in ("complete_case_days", "excluded_days"):
        if type(quality[field]) is not int or quality[field] < 0:
            raise DataError(f"quality report {field} must be a nonnegative integer")
    if quality["complete_case_days"] + quality["excluded_days"] != expected_days:
        raise DataError("quality report complete and excluded days do not match calendar_days")
    if (
        not isinstance(quality["excluded_dates"], list)
        or len(quality["excluded_dates"]) != quality["excluded_days"]
    ):
        raise DataError("quality report excluded_dates does not match excluded_days")
    if not isinstance(quality["exclusion_policy"], str) or not quality["exclusion_policy"]:
        raise DataError("quality report exclusion_policy must be a nonempty string")

    if not paths.raw_path.is_file():
        raise DataError(f"raw source data is missing at {paths.raw_path}; run fetch first")
    try:
        raw_bytes = paths.raw_path.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read raw source data: {error}") from error
    raw_digest = hashlib.sha256(raw_bytes).hexdigest()
    digests = (sidecar["sha256"], quality["raw_sha256"])
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in digests
    ):
        raise DataError("raw source checksum metadata must contain SHA-256 digests")
    if not (sidecar["sha256"].lower() == quality["raw_sha256"].lower() == raw_digest):
        raise DataError("raw response bytes do not match sidecar and quality report checksums")

    if not paths.daily_path.is_file():
        raise DataError(f"prepared daily data is missing at {paths.daily_path}; run prepare first")
    try:
        prepared_bytes = paths.daily_path.read_bytes()
    except OSError as error:
        raise DataError(f"cannot read prepared daily data: {error}") from error
    prepared_digest = quality["prepared_daily_csv_sha256"]
    if not isinstance(prepared_digest, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", prepared_digest
    ):
        raise DataError("prepared daily checksum must be a SHA-256 digest")
    if hashlib.sha256(prepared_bytes).hexdigest() != prepared_digest.lower():
        raise DataError("prepared daily CSV bytes do not match the quality report checksum")
    return quality, prepared_bytes


def load_development_daily(paths: ProjectPaths, config: ExperimentConfig) -> pd.DataFrame:
    """Load verified data through validation; never parse final-test rows."""
    if not paths.daily_path.is_file():
        raise DataError(
            f"prepared daily data is missing at {paths.daily_path}; run fetch and prepare first"
        )
    _, prepared_bytes = _validate_development_provenance(paths, config)

    development_days = (config.validation_end - config.start_date).days + 1
    if development_days <= 0:
        raise DataError("validation end date must follow the experiment start date")
    try:
        data = pd.read_csv(io.BytesIO(prepared_bytes), nrows=development_days, parse_dates=["date"])
    except (ValueError, pd.errors.ParserError) as error:
        raise DataError(f"cannot read prepared development data: {error}") from error
    expected_columns = ["date", *config.parameters]
    if list(data.columns) != expected_columns:
        raise DataError(
            "prepared development CSV columns do not match the configured schema: "
            f"expected {expected_columns}, found {list(data.columns)}"
        )
    expected = pd.date_range(config.start_date, config.validation_end, freq="D")
    if len(data) != len(expected):
        raise DataError(
            f"prepared development data has {len(data)} rows; expected {len(expected)} "
            "through the validation boundary"
        )
    try:
        dates = pd.DatetimeIndex(data["date"])
    except (TypeError, ValueError) as error:
        raise DataError("prepared development dates contain invalid values") from error
    if not dates.equals(expected):
        raise DataError("prepared development rows are not a complete sorted daily calendar")
    return data.set_index("date")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_daily_frame(daily: pd.DataFrame) -> str:
    payload = (
        daily.rename_axis("date").to_csv(index_label="date", lineterminator="\n").encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_hash(values: Any) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _code_hashes() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    paths = {
        "config": repository / "src" / "sunreserve" / "config.py",
        "data": repository / "src" / "sunreserve" / "data.py",
        "features": repository / "src" / "sunreserve" / "features.py",
        "decision": repository / "src" / "sunreserve" / "decision.py",
        "train": repository / "src" / "sunreserve" / "train.py",
        "evaluate": repository / "src" / "sunreserve" / "evaluate.py",
        "cli": repository / "src" / "sunreserve" / "__main__.py",
        "inference": repository / "src" / "sunreserve" / "inference.py",
        "source_provenance": repository / "src" / "sunreserve" / "source_provenance.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with open(descriptor, "w", encoding="utf-8", newline="\n", closefd=True) as stream:
            stream.write(content)
            stream.flush()
        Path(temporary).replace(path)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


def run_validation(
    paths: ProjectPaths,
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
    model_path: Path | None = None,
    recorded_model_path: Path | None = None,
    _verified_daily: pd.DataFrame | None = None,
) -> ValidationSelection:
    """Run cached validation-only selection and write versioned development artifacts."""
    daily = (
        _verified_daily if _verified_daily is not None else load_development_daily(paths, config)
    )
    rows, audit = build_supervised_rows(
        daily,
        config,
        target_date_end=config.validation_end,
    )
    partitions = {
        name: rows.loc[rows["split"] == name].copy().reset_index(drop=True)
        for name in ("train", "validation")
    }
    if partitions["train"].empty:
        raise RuntimeError("training rows unexpectedly include an empty partition")
    if partitions["validation"].empty:
        raise DataError("no eligible validation rows remain after feature exclusions")
    if set(rows["split"]) - {"train", "validation"}:
        raise RuntimeError("validation rows unexpectedly include the final test partition")

    selection = select_validation_model(partitions["train"], partitions["validation"], config)
    model_output = model_path or paths.model_path
    validation_output = output_dir or paths.validation_dir
    model_bundle, model_bundle_sha256 = write_model_bundle(
        partitions["train"], config, model_output
    )
    selection = ValidationSelection(
        selected_model=selection.selected_model,
        low_solar_threshold=selection.low_solar_threshold,
        metrics=selection.metrics,
        predictions=selection.predictions,
        eligible_rows={
            "train": int(len(partitions["train"])),
            "validation": int(len(partitions["validation"])),
        },
        excluded_dates=audit["excluded_dates"],
    )
    decision_results = evaluate_decision_scenarios(
        selection.predictions,
        config,
        selected_model=selection.selected_model,
    )
    primary_decisions = decision_results.metrics.loc[
        (decision_results.metrics["scenario_id"].astype(str) == "primary")
        & decision_results.metrics["approach"].isin(DECISION_SELECTION_TIE_ORDER)
    ].copy()
    if set(primary_decisions["approach"]) != set(DECISION_SELECTION_TIE_ORDER):
        raise RuntimeError("primary validation decision metrics omit a candidate approach")
    decision_scores = {
        str(row.approach): float(
            row.mean_simulated_shortfall_kwh_day + row.mean_excess_prepared_reserve_kwh_day
        )
        for row in primary_decisions.itertuples(index=False)
    }
    decision_selected_model = min(
        DECISION_SELECTION_TIE_ORDER,
        key=lambda name: (decision_scores[name], DECISION_SELECTION_TIE_ORDER.index(name)),
    )

    input_hashes = {
        "raw_response_sha256": json.loads(paths.sidecar_path.read_text(encoding="utf-8"))["sha256"],
        "quality_report_sha256": _sha256_file(paths.quality_path),
        "prepared_daily_csv_sha256": json.loads(paths.quality_path.read_text(encoding="utf-8"))[
            "prepared_daily_csv_sha256"
        ],
        "prepared_development_rows_sha256": _sha256_daily_frame(daily),
    }
    config_values = config.to_dict()
    config_hash = _canonical_json_hash(config_values)
    code_hashes = _code_hashes()
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "sunreserve_validation_selection",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "validation_only; final test not loaded, predicted, scored, or reported",
        "configuration": config_values,
        "configuration_sha256": config_hash,
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        **source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "random_seed": config.random_seed,
        "training_target_period": [
            config.start_date.isoformat(),
            config.training_end.isoformat(),
        ],
        "validation_target_period": [
            config.validation_start.isoformat(),
            config.validation_end.isoformat(),
        ],
        "eligible_rows": selection.eligible_rows,
        "excluded_rows": len(selection.excluded_dates),
        "excluded_dates": selection.excluded_dates,
        "feature_schema": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "target_units": "kWh/m^2/day",
        "ridge": {
            "alpha": config.ridge_alpha,
            "solver": "lsqr",
            "tolerance": 1e-4,
            "max_iterations": 1000,
            "random_state": config.random_seed,
            "preprocessing": [
                "median imputation fitted on training rows",
                "standard scaling fitted on training rows",
            ],
        },
        "model_selection": {
            "metric": config.primary_metric,
            "selection_partition": "validation",
            "tie_order": list(SELECTION_TIE_ORDER),
            "selected_model": selection.selected_model,
            "validation_metrics": selection.metrics,
            "prediction_clip_rule": PREDICTION_CLIP_RULE,
            "training_only_low_solar_threshold_kwh_m2_day": selection.low_solar_threshold,
            "low_solar_quantile": config.low_solar_quantile,
            "quantile_method": "numpy linear",
            "validation_refit": False,
        },
        "decision_selection": {
            "objective": DECISION_SELECTION_OBJECTIVE,
            "score_name": "mean_shortfall_plus_mean_excess_kwh_day",
            "shortfall_weight": 1.0,
            "excess_prepared_reserve_weight": 1.0,
            "selection_partition": "validation",
            "selection_target_period": [
                config.validation_start.isoformat(),
                config.validation_end.isoformat(),
            ],
            "scenario_id": "primary",
            "tie_order": list(DECISION_SELECTION_TIE_ORDER),
            "scores": decision_scores,
            "selected_model": decision_selected_model,
            "interpretation": (
                "Research ranking only. Equal energy weights are hypothetical and do not "
                "represent costs, reliability guarantees, or an operational policy."
            ),
        },
        "model_bundle": {
            "path": (recorded_model_path or model_output).relative_to(paths.root).as_posix(),
            "sha256": model_bundle_sha256,
            "model_id": model_bundle.model_id,
            "training_split": "train",
            "training_rows": model_bundle.training_rows,
            "training_target_period": [
                model_bundle.training_start.isoformat(),
                model_bundle.training_end.isoformat(),
            ],
        },
        "decision_simulation": {
            "partition": "validation",
            "generation_proxy": ("radiation_kwh_m2_day * panel_area_m2 * efficiency * derating"),
            "prepared_reserve": "min(capacity, max(0, demand - predicted_generation + buffer))",
            "simulated_shortfall": "max(0, demand - actual_generation - prepared_reserve)",
            "additional_shortfall_above_capacity_floor": (
                "max(0, simulated_shortfall - capacity_limited_shortfall); descriptive, "
                "not a pure causal forecast-error estimate"
            ),
            "excess_prepared_reserve": (
                "max(0, prepared_reserve - max(0, demand - actual_generation))"
            ),
            "capacity_limited_shortfall": ("max(0, demand - actual_generation - capacity)"),
            "primary_scenario": {
                "panel_area_m2": config.panel_area_m2,
                "efficiency": config.conversion_efficiency,
                "derating": config.derating_factor,
                "demand_kwh": config.daily_demand_kwh,
                "capacity_kwh": config.reserve_cap_kwh,
                "buffer_kwh": config.reserve_buffer_kwh,
            },
            "sensitivity_grid": {
                "demand_kwh": list(DEMAND_LEVELS_KWH),
                "capacity_kwh": list(CAPACITY_LEVELS_KWH),
                "buffer_fraction_of_demand": list(BUFFER_FRACTIONS),
                "scenario_count": 27,
            },
            "approaches": ["persistence", "seasonal_mean", "ridge", "always_prepare_backup"],
            "paired_differences": (
                "same validation target dates; mean daily simulated shortfall and excess "
                "versus persistence and seasonal_mean"
            ),
            "artifacts": ["decision_predictions.csv", "decision_metrics.csv"],
            "metrics": decision_results.metrics.to_dict(orient="records"),
        },
    }
    predictions_content = selection.predictions.to_csv(index=False, lineterminator="\n")
    metrics_content = (
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "selection_metric": config.primary_metric,
                "validation_metrics": selection.metrics,
                "selected_model": selection.selected_model,
                "training_only_low_solar_threshold_kwh_m2_day": selection.low_solar_threshold,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    decision_predictions_content = decision_results.predictions.to_csv(
        index=False, lineterminator="\n"
    )
    decision_metrics_content = decision_results.metrics.to_csv(index=False, lineterminator="\n")
    development_artifacts = {
        "validation_predictions.csv": predictions_content,
        "validation_metrics.json": metrics_content,
        "decision_predictions.csv": decision_predictions_content,
        "decision_metrics.csv": decision_metrics_content,
    }
    manifest["development_artifact_hashes"] = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in development_artifacts.items()
    }
    manifest_content = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write_text(validation_output / "validation_predictions.csv", predictions_content)
    _atomic_write_text(validation_output / "validation_metrics.json", metrics_content)
    _atomic_write_text(validation_output / "selection_manifest.json", manifest_content)
    _atomic_write_text(validation_output / "decision_predictions.csv", decision_predictions_content)
    _atomic_write_text(validation_output / "decision_metrics.csv", decision_metrics_content)
    return selection


def run_experiment_validation(
    paths: ProjectPaths, config: ExperimentConfig, run_id: str
) -> tuple[ValidationSelection, Path]:
    """Publish an immutable, validation-only experiment directory atomically."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("--run-id must be 1-64 safe letters, digits, hyphens, or underscores")
    experiments = paths.root / "artifacts" / "experiments"
    locks = experiments / ".locks"
    reservation = locks / f"{run_id}.lock"
    destination = experiments / run_id
    if destination.exists():
        raise FileExistsError(f"experiment already exists and is immutable: {destination}")
    if reservation.exists():
        raise FileExistsError(
            f"experiment ID is reserved (active or stale); inspect and resolve manually: {run_id}"
        )

    verified_daily = load_development_daily(paths, config)
    experiments.mkdir(parents=True, exist_ok=True)
    locks.mkdir(exist_ok=True)
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"experiment ID is reserved (active or stale); inspect and resolve manually: {run_id}"
        ) from error
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=experiments))
        validation_dir = staging / "validation"
        model_path = staging / "models" / "ridge_model.json"
        validation_dir.mkdir(parents=True)
        model_path.parent.mkdir(parents=True)
        selection = run_validation(
            paths,
            config,
            output_dir=validation_dir,
            model_path=model_path,
            recorded_model_path=destination / "models" / "ridge_model.json",
            _verified_daily=verified_daily,
        )
        manifest_path = validation_dir / "selection_manifest.json"
        selection_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_manifest = {
            "schema_version": 1,
            "artifact_type": "sunreserve_validation_experiment",
            **source_provenance(),
            "run_id": run_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "selected_model": selection.selected_model,
            "selected_approach": selection_manifest["decision_selection"]["selected_model"],
            "model_path": (destination / "models" / "ridge_model.json")
            .relative_to(paths.root)
            .as_posix(),
            "model_sha256": _sha256_file(model_path),
            "selection_manifest_sha256": _sha256_file(manifest_path),
            "configuration": config.to_dict(),
            "configuration_sha256": _canonical_json_hash(config.to_dict()),
        }
        (staging / "experiment_manifest.json").write_text(
            json.dumps(experiment_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError(f"experiment already exists and is immutable: {destination}")
        staging.rename(destination)
        return selection, destination / "models" / "ridge_model.json"
    except Exception:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        reservation.rmdir()


@dataclass(frozen=True)
class FinalizedRun:
    """Identity and test-row counts for one immutable finalized experiment run."""

    run_id: str
    run_dir: Path
    row_counts: dict[str, int]
    selected_model: str


def _read_finalized_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise DataError(f"{label} is unreadable or contains invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise DataError(f"{label} must contain a JSON object")
    return result


def _finalized_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as error:
        raise DataError(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise DataError(f"{label} must be finite")
    if nonnegative and number < 0:
        raise DataError(f"{label} must be nonnegative")
    return number


def _require_finalized_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    if frame.columns.has_duplicates or set(frame.columns) != expected:
        raise DataError(f"{label} columns do not match the finalized schema")


def _finalized_dates(series: pd.Series, label: str) -> pd.DatetimeIndex:
    values = series.tolist()
    if any(
        not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        for value in values
    ):
        raise DataError(f"{label} contains an invalid date")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, format="%Y-%m-%d", errors="raise"))
    except (TypeError, ValueError, OverflowError) as error:
        raise DataError(f"{label} contains an invalid date: {error}") from error
    if dates.isna().any():
        raise DataError(f"{label} contains an invalid date")
    return dates


def _compare_finalized_json(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise DataError(f"{label} has an invalid schema")
        for key, expected_value in expected.items():
            _compare_finalized_json(actual[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, float):
        actual_number = _finalized_number(actual, label)
        if not math.isclose(actual_number, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise DataError(f"{label} does not match the finalized predictions")
        return
    if isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise DataError(f"{label} does not match the finalized predictions")
        return
    if actual != expected:
        raise DataError(f"{label} does not match the finalized predictions")


def _validate_finalized_forecast_metrics(
    payload: dict[str, Any], prediction_frame: pd.DataFrame, selected_model: str
) -> None:
    expected_top_fields = {
        "schema_version",
        "partition",
        "target_units",
        "selected_model_frozen_on_validation",
        "training_only_low_solar_threshold_kwh_m2_day",
        "metrics",
    }
    if set(payload) != expected_top_fields:
        raise DataError("forecast metrics have an invalid schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != FINAL_RUN_SCHEMA_VERSION
        or payload["partition"] != "test"
        or payload["target_units"] != "kWh/m^2/day"
        or payload["selected_model_frozen_on_validation"] != selected_model
    ):
        raise DataError("forecast metrics schema, partition, units, or selected model are invalid")
    threshold = _finalized_number(
        payload["training_only_low_solar_threshold_kwh_m2_day"],
        "forecast low-solar threshold",
        nonnegative=True,
    )
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != FINAL_FORECAST_APPROACHES:
        raise DataError("forecast metrics must cover all supported approaches")
    actual = prediction_frame["actual_radiation_kwh_m2_day"].to_numpy(dtype=float)
    date_values = pd.DatetimeIndex(prediction_frame["target_date"])
    columns = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }
    expected_metrics = {
        approach: forecast_metrics(
            actual,
            prediction_frame[column].to_numpy(dtype=float),
            date_values,
            low_solar_threshold=threshold,
        )
        for approach, column in columns.items()
    }
    for approach in FINAL_FORECAST_APPROACHES:
        metric = metrics[approach]
        if not isinstance(metric, dict) or set(metric) != FINAL_FORECAST_METRIC_FIELDS:
            raise DataError(f"forecast metrics for {approach} have an invalid schema")
        _compare_finalized_json(metric, expected_metrics[approach], f"forecast metrics {approach}")


def _require_finalized_numeric_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
    *,
    nonnegative: set[str] | None = None,
    allow_nan_by_approach: set[str] | None = None,
) -> None:
    nonnegative = nonnegative or set()
    allow_nan_by_approach = allow_nan_by_approach or set()
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise DataError(f"{label} field {column} must be numeric") from error
        missing = np.isnan(values)
        if column in allow_nan_by_approach:
            permitted_missing = frame["approach"].eq("always_prepare_backup").to_numpy()
            if (missing & ~permitted_missing).any() or np.isinf(values).any():
                raise DataError(f"{label} field {column} contains non-finite values")
            finite = values[~missing]
        else:
            if not np.isfinite(values).all():
                raise DataError(f"{label} field {column} contains non-finite values")
            finite = values
        if column in nonnegative and (finite < 0).any():
            raise DataError(f"{label} field {column} must be nonnegative")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DataError(f"{label} is unreadable or contains invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DataError(f"{label} must contain a JSON object")
    return value


def _validate_frozen_development(
    paths: ProjectPaths,
    expected_manifest_sha256: str,
    expected_config: ExperimentConfig | None,
) -> tuple[dict[str, Any], ExperimentConfig, dict[str, Any], pd.DataFrame]:
    manifest_path = paths.validation_dir / "selection_manifest.json"
    if not manifest_path.is_file():
        raise DataError(f"selection manifest is missing: {manifest_path}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_manifest_sha256):
        raise ValueError("--manifest-sha256 must be a 64-character SHA-256 hex digest")
    actual_manifest_hash = _sha256_file(manifest_path)
    if actual_manifest_hash.lower() != expected_manifest_sha256.lower():
        raise DataError("selection manifest SHA-256 does not match the explicitly frozen digest")

    manifest = _read_json_object(manifest_path, "selection manifest")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("artifact_type") != "sunreserve_validation_selection"
        or not str(manifest.get("status", "")).startswith("validation_only")
    ):
        raise DataError(
            "selection manifest schema, type, or validation-only status is incompatible"
        )
    try:
        config = ExperimentConfig.from_dict(manifest["configuration"])
    except (KeyError, TypeError, ValueError) as error:
        raise DataError(f"selection manifest configuration is invalid: {error}") from error
    if manifest.get("configuration_sha256") != _canonical_json_hash(config.to_dict()):
        raise DataError("selection manifest configuration hash does not match its content")
    if expected_config is not None and expected_config.to_dict() != config.to_dict():
        raise DataError("active configuration does not match the frozen selection manifest")
    if manifest.get("code_hashes") != _code_hashes():
        raise DataError("current experiment code does not match the frozen selection manifest")
    bundle_manifest = manifest.get("model_bundle", {})
    if bundle_manifest.get("path") != paths.model_path.relative_to(paths.root).as_posix():
        raise DataError("frozen model bundle path is missing or incompatible")
    if not paths.model_path.is_file() or _sha256_file(paths.model_path) != bundle_manifest.get(
        "sha256"
    ):
        raise DataError("trained model bundle is missing or differs from the frozen manifest")
    if bundle_manifest.get("training_split") != "train":
        raise DataError("trained model bundle is not declared as training-only")
    bundle_manifest = manifest.get("model_bundle", {})
    if bundle_manifest.get("path") != paths.model_path.relative_to(paths.root).as_posix():
        raise DataError("frozen model bundle path is missing or incompatible")
    if not paths.model_path.is_file() or _sha256_file(paths.model_path) != bundle_manifest.get(
        "sha256"
    ):
        raise DataError("trained model bundle is missing or differs from the frozen manifest")
    if bundle_manifest.get("training_split") != "train":
        raise DataError("trained model bundle is not declared as training-only")
    selected = manifest.get("model_selection", {}).get("selected_model")
    if selected not in SELECTION_TIE_ORDER:
        raise DataError("frozen manifest does not select a supported validation candidate")
    if manifest.get("validation_target_period") != [
        config.validation_start.isoformat(),
        config.validation_end.isoformat(),
    ]:
        raise DataError("frozen validation period does not match the saved config")

    quality = _read_json_object(paths.quality_path, "data quality report")
    provenance = _read_json_object(paths.sidecar_path, "raw provenance sidecar")
    input_hashes = manifest.get("input_hashes", {})
    if not paths.raw_path.is_file() or _sha256_file(paths.raw_path) != provenance.get("sha256"):
        raise DataError("raw source cache checksum does not match its provenance sidecar")
    if provenance.get("sha256") != quality.get("raw_sha256"):
        raise DataError("quality report and source sidecar refer to different raw responses")
    if input_hashes.get("quality_report_sha256") != _sha256_file(paths.quality_path):
        raise DataError("quality report changed after the validation manifest was frozen")
    if input_hashes.get("raw_response_sha256") != provenance.get("sha256"):
        raise DataError("frozen run refers to a different raw source response")
    if provenance.get("request_url") != request_url(config):
        raise DataError("raw source request does not match the frozen configuration")
    prepared_hash = quality.get("prepared_daily_csv_sha256")
    if not prepared_hash or input_hashes.get("prepared_daily_csv_sha256") != prepared_hash:
        raise DataError("prepared daily checksum is missing or differs from the frozen manifest")
    if (
        quality.get("requested_start") != config.start_date.isoformat()
        or quality.get("requested_end") != config.end_date.isoformat()
    ):
        raise DataError("prepared data range does not match the frozen configuration")

    # Check the complete prepared file by checksum before inspecting any test rows.
    # The file is hashed as bytes only; CSV parsing remains behind all freeze checks.
    if _sha256_file(paths.daily_path) != prepared_hash:
        raise DataError("prepared daily file hash differs from its quality report")

    required_artifact_hashes = {
        "validation_predictions.csv",
        "validation_metrics.json",
        "decision_predictions.csv",
        "decision_metrics.csv",
    }
    artifact_hashes = manifest.get("development_artifact_hashes", {})
    if set(artifact_hashes) != required_artifact_hashes:
        raise DataError("frozen manifest does not contain all development artifact hashes")
    for name, expected_hash in artifact_hashes.items():
        path = paths.validation_dir / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise DataError(f"development artifact is missing or changed: {name}")

    daily = load_development_daily(paths, config)
    if _sha256_daily_frame(daily) != input_hashes.get("prepared_development_rows_sha256"):
        raise DataError("prepared training/validation rows differ from the frozen manifest")
    # The exact manifest digest and all source/prepared/development inputs/code were
    # verified above before the finalizer parses the full daily file.
    return manifest, config, quality, daily


def _validate_experiment_development(
    paths: ProjectPaths,
    experiment_id: str,
    expected_manifest_sha256: str,
    expected_config: ExperimentConfig | None,
) -> tuple[dict[str, Any], ExperimentConfig, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Verify one experiment's identity and artifacts before delegating freeze checks."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", experiment_id):
        raise ValueError("experiment_id must be 1-64 safe letters, digits, hyphens, or underscores")
    experiment_dir = paths.root / "artifacts" / "experiments" / experiment_id
    experiments_dir = paths.root / "artifacts" / "experiments"
    if experiment_dir.resolve().parent != experiments_dir.resolve():
        raise DataError("experiment directory resolves outside the experiment artifact root")
    manifest_path = experiment_dir / "experiment_manifest.json"
    experiment = _read_json_object(manifest_path, "experiment manifest")
    if (
        experiment.get("schema_version") != 1
        or experiment.get("artifact_type") != "sunreserve_validation_experiment"
        or experiment.get("run_id") != experiment_id
    ):
        raise DataError("experiment manifest schema, type, or identifier is incompatible")
    try:
        config = ExperimentConfig.from_dict(experiment["configuration"])
    except (KeyError, TypeError, ValueError) as error:
        raise DataError(f"experiment configuration is invalid: {error}") from error
    config_digest = _canonical_json_hash(config.to_dict())
    if (
        experiment["configuration"] != config.to_dict()
        or experiment.get("configuration_sha256") != config_digest
    ):
        raise DataError("experiment configuration digest does not match its content")
    if expected_config is not None and expected_config.to_dict() != config.to_dict():
        raise DataError("active configuration does not match the selected experiment")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_manifest_sha256):
        raise ValueError("--manifest-sha256 must be a 64-character SHA-256 hex digest")
    selection_path = experiment_dir / "validation" / "selection_manifest.json"
    expected_selection_path = f"artifacts/experiments/{experiment_id}/models/ridge_model.json"
    if experiment.get("model_path") != expected_selection_path:
        raise DataError("experiment model path is missing or incompatible")
    actual_selection_digest = _sha256_file(selection_path) if selection_path.is_file() else None
    model_path = experiment_dir / "models" / "ridge_model.json"
    if (
        (experiment_dir / "validation").resolve().parent != experiment_dir.resolve()
        or (experiment_dir / "models").resolve().parent != experiment_dir.resolve()
        or selection_path.is_symlink()
        or model_path.is_symlink()
    ):
        raise DataError("experiment validation or model artifact resolves outside its directory")
    if (
        actual_selection_digest is None
        or actual_selection_digest.lower() != expected_manifest_sha256.lower()
        or actual_selection_digest.lower()
        != str(experiment.get("selection_manifest_sha256", "")).lower()
    ):
        raise DataError("selection manifest SHA-256 does not match the frozen experiment digest")
    if not model_path.is_file() or _sha256_file(model_path) != experiment.get("model_sha256"):
        raise DataError("experiment model is missing or differs from its recorded checksum")
    selection = _read_json_object(selection_path, "selection manifest")
    bundle = selection.get("model_bundle", {})
    if (
        bundle.get("path") != expected_selection_path
        or bundle.get("sha256") != experiment.get("model_sha256")
        or selection.get("configuration_sha256") != config_digest
        or experiment.get("selected_model")
        != selection.get("model_selection", {}).get("selected_model")
        or experiment.get("selected_approach")
        != selection.get("decision_selection", {}).get("selected_model")
    ):
        raise DataError(
            "selection manifest model or configuration identity differs from experiment"
        )
    scoped_paths = copy(paths)
    scoped_paths.validation_dir = experiment_dir / "validation"
    scoped_paths.model_path = model_path
    manifest, verified_config, quality, development = _validate_frozen_development(
        scoped_paths,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_config=expected_config,
    )
    return manifest, verified_config, quality, development, experiment


def _load_finalization_model_bundle(
    paths: ProjectPaths,
    selection_manifest: dict[str, Any],
    source_experiment: dict[str, Any] | None,
) -> tuple[RidgeModelBundle, dict[str, Any]]:
    """Load the exact frozen numeric model after source identity checks pass."""
    bundle_manifest = selection_manifest.get("model_bundle")
    if not isinstance(bundle_manifest, dict):
        raise DataError("frozen selection manifest has no model bundle identity")
    if source_experiment is None:
        expected_relative_path = paths.model_path.relative_to(paths.root).as_posix()
    else:
        expected_relative_path = source_experiment.get("model_path")
        if not isinstance(expected_relative_path, str):
            raise DataError("frozen experiment has no model bundle path")
    relative_path = bundle_manifest.get("path")
    if relative_path != expected_relative_path or not isinstance(relative_path, str):
        raise DataError("frozen model bundle path differs from its validated source")
    path_parts = relative_path.split("/")
    if "\\" in relative_path or any(part in {"", ".", ".."} for part in path_parts):
        raise DataError("frozen model bundle path is invalid")
    model_path = paths.root.joinpath(*path_parts)
    try:
        model_path.resolve().relative_to(paths.root.resolve())
    except ValueError as error:
        raise DataError("frozen model bundle resolves outside the project root") from error

    expected_sha256 = bundle_manifest.get("sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_sha256
    ):
        raise DataError("frozen model bundle checksum is invalid")
    try:
        bundle = load_model_bundle(model_path, expected_sha256=expected_sha256)
    except ModelArtifactError as error:
        raise DataError(f"frozen model bundle is invalid: {error}") from error

    expected_period = [bundle.training_start.isoformat(), bundle.training_end.isoformat()]
    if (
        bundle_manifest.get("model_id") != bundle.model_id
        or bundle_manifest.get("training_split") != "train"
        or type(bundle_manifest.get("training_rows")) is not int
        or bundle_manifest.get("training_rows") != bundle.training_rows
        or bundle_manifest.get("training_target_period") != expected_period
    ):
        raise DataError("frozen model bundle metadata does not match its numeric model")
    identity = {
        "path": relative_path,
        "model_id": bundle.model_id,
        "sha256": bundle.artifact_sha256,
        "training_split": "train",
        "training_rows": bundle.training_rows,
        "training_target_period": expected_period,
    }
    return bundle, identity


def _load_full_daily(paths: ProjectPaths, config: ExperimentConfig) -> pd.DataFrame:
    expected = pd.date_range(config.start_date, config.end_date, freq="D")
    try:
        full = pd.read_csv(paths.daily_path, parse_dates=["date"])
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataError(f"cannot read prepared full daily data: {error}") from error
    if not {"date", *config.parameters} <= set(full.columns):
        raise DataError("prepared full daily data has an incompatible schema")
    dates = pd.DatetimeIndex(full.pop("date"))
    if not dates.equals(expected):
        raise DataError("prepared full data is not the configured sorted daily calendar")
    full.index = dates
    full.index.name = "date"
    return full


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _safe_run_id(run_id: str | None) -> str:
    if run_id is None:
        return datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("run_id must be 1–64 safe letters, numbers, dots, underscores, or dashes")
    return run_id


def finalize_run(
    paths: ProjectPaths,
    *,
    manifest_sha256: str,
    run_id: str | None = None,
    expected_config: ExperimentConfig | None = None,
    experiment_id: str | None = None,
) -> FinalizedRun:
    """Evaluate the held-out test only after an explicitly frozen, verified manifest."""
    safe_id = _safe_run_id(run_id)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = paths.runs_dir / safe_id
    if run_dir.exists():
        raise FileExistsError(
            f"finalized run directory already exists and will not be overwritten: {run_dir}"
        )

    source_experiment = None
    selection_source_dir = paths.validation_dir
    if experiment_id is None:
        manifest, config, quality, _development = _validate_frozen_development(
            paths,
            expected_manifest_sha256=manifest_sha256,
            expected_config=expected_config,
        )
    else:
        manifest, config, quality, _development, source_experiment = (
            _validate_experiment_development(
                paths,
                experiment_id,
                manifest_sha256,
                expected_config,
            )
        )
        selection_source_dir = (
            paths.root / "artifacts" / "experiments" / experiment_id / "validation"
        )
    model_bundle, model_bundle_identity = _load_finalization_model_bundle(
        paths, manifest, source_experiment
    )
    daily = _load_full_daily(paths, config)
    rows, split_audit = build_supervised_rows(daily, config)
    excluded_dates_by_split: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("train", "validation", "test")
    }
    for excluded in split_audit["excluded_dates"]:
        split_name = _split_name(pd.Timestamp(excluded["target_date"]), config)
        excluded_dates_by_split[split_name].append(excluded)
    excluded_rows_by_split = {
        name: len(excluded_dates) for name, excluded_dates in excluded_dates_by_split.items()
    }
    partitions = split_supervised_rows(rows)
    training = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    if training.empty or validation.empty or test.empty:
        raise DataError("finalization requires nonempty train, validation, and test partitions")
    if test["target_date"].dt.date.min() < config.test_start:
        raise DataError("final test rows include a target date before configured test_start")

    threshold = float(
        np.quantile(
            pd.to_numeric(training[TARGET_COLUMN], errors="raise").to_numpy(dtype=float),
            config.low_solar_quantile,
            method="linear",
        )
    )
    frozen_threshold = manifest["model_selection"].get(
        "training_only_low_solar_threshold_kwh_m2_day"
    )
    if frozen_threshold is None or not np.isclose(
        threshold, float(frozen_threshold), rtol=0, atol=1e-12
    ):
        raise DataError("training-only low-solar threshold does not match the frozen manifest")

    predictions = test.loc[:, ["issue_date", "target_date", TARGET_COLUMN]].copy()
    predictions["persistence_prediction_kwh_m2_day"] = predict_persistence(test).to_numpy()
    predictions["seasonal_mean_prediction_kwh_m2_day"] = predict_seasonal_mean_bundle(
        model_bundle, test
    ).to_numpy()
    predictions["ridge_prediction_kwh_m2_day"] = predict_ridge_bundle(model_bundle, test).to_numpy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual_radiation_kwh_m2_day"})
    predictions["split"] = "test"
    predictions["selected_model"] = manifest["model_selection"]["selected_model"]
    predictions = predictions[
        [
            "issue_date",
            "target_date",
            "split",
            "actual_radiation_kwh_m2_day",
            "persistence_prediction_kwh_m2_day",
            "seasonal_mean_prediction_kwh_m2_day",
            "ridge_prediction_kwh_m2_day",
            "selected_model",
        ]
    ]
    if not predictions["target_date"].equals(test["target_date"]):
        raise DataError("final test forecasts do not share identical target dates")
    prediction_arrays = predictions[
        [
            "actual_radiation_kwh_m2_day",
            "persistence_prediction_kwh_m2_day",
            "seasonal_mean_prediction_kwh_m2_day",
            "ridge_prediction_kwh_m2_day",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(prediction_arrays).all():
        raise DataError("final predictions contain non-finite values")

    prediction_columns = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }
    test_actual = predictions["actual_radiation_kwh_m2_day"].to_numpy(dtype=float)
    test_dates = predictions["target_date"]
    forecast_results = {
        approach: forecast_metrics(
            test_actual,
            predictions[column],
            test_dates,
            low_solar_threshold=threshold,
        )
        for approach, column in prediction_columns.items()
    }
    forecast_payload = {
        "schema_version": FINAL_RUN_SCHEMA_VERSION,
        "partition": "test",
        "target_units": "kWh/m^2/day",
        "selected_model_frozen_on_validation": manifest["model_selection"]["selected_model"],
        "training_only_low_solar_threshold_kwh_m2_day": threshold,
        "metrics": forecast_results,
    }
    decision_input = predictions.rename(
        columns={
            "actual_radiation_kwh_m2_day": RADIATION_COLUMN,
            **{
                column: f"{approach}_prediction_kwh_m2_day"
                for approach, column in prediction_columns.items()
            },
        }
    )
    decision_input = decision_input.rename(
        columns={
            "persistence_prediction_kwh_m2_day": "persistence_prediction_kwh_m2_day",
            "seasonal_mean_prediction_kwh_m2_day": "seasonal_mean_prediction_kwh_m2_day",
            "ridge_prediction_kwh_m2_day": "ridge_prediction_kwh_m2_day",
        }
    )
    decision_results = evaluate_decision_scenarios(
        decision_input,
        config,
        selected_model=manifest["model_selection"]["selected_model"],
    )
    decision_predictions = decision_results.predictions.copy()
    scenario_settings = decision_predictions[
        ["scenario_id", "approach", "demand_kwh", "capacity_kwh", "buffer_kwh"]
    ].drop_duplicates()
    decision_metrics = decision_results.metrics.merge(
        scenario_settings,
        on=["scenario_id", "approach"],
        how="left",
        validate="one_to_one",
    )
    decision_predictions.insert(0, "artifact_schema_version", FINAL_RUN_SCHEMA_VERSION)
    decision_predictions.insert(1, "split", "test")
    decision_metrics.insert(0, "artifact_schema_version", FINAL_RUN_SCHEMA_VERSION)
    decision_metrics.insert(1, "partition", "test")
    predictions.insert(0, "artifact_schema_version", FINAL_RUN_SCHEMA_VERSION)

    selection_manifest_bytes = (selection_source_dir / "selection_manifest.json").read_bytes()
    assumptions_path = paths.root / "docs" / "assumptions.md"
    if not assumptions_path.is_file():
        raise DataError(f"assumptions documentation is missing: {assumptions_path}")
    assumptions_text = assumptions_path.read_text(encoding="utf-8")
    payloads = {
        "predictions.csv": predictions.to_csv(index=False, lineterminator="\n"),
        "forecast_metrics.json": _json_text(forecast_payload),
        "decision_metrics.csv": decision_metrics.to_csv(index=False, lineterminator="\n"),
        "decision_predictions.csv": decision_predictions.to_csv(index=False, lineterminator="\n"),
        "data_quality.json": _json_text(quality),
        "selection_manifest.json": selection_manifest_bytes.decode("utf-8"),
        "assumptions.md": assumptions_text,
    }
    artifact_hashes = {name: _sha256_text(value) for name, value in payloads.items()}
    run_manifest = {
        "schema_version": FINAL_RUN_SCHEMA_VERSION,
        "artifact_type": (
            "sunreserve_finalized_experiment"
            if source_experiment is not None
            else "sunreserve_finalized_historical_replay"
        ),
        "run_id": safe_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "partition": "test",
        "test_inspected": True,
        "test_inspected_utc": datetime.now(UTC).isoformat(),
        "selected_model_frozen_on_validation": manifest["model_selection"]["selected_model"],
        "model_bundle": model_bundle_identity,
        "selection_manifest_sha256": manifest_sha256.lower(),
        "configuration_sha256": manifest["configuration_sha256"],
        "source_checksums": {
            "raw_response_sha256": manifest["input_hashes"]["raw_response_sha256"],
            "quality_report_sha256": manifest["input_hashes"]["quality_report_sha256"],
            "prepared_daily_csv_sha256": manifest["input_hashes"]["prepared_daily_csv_sha256"],
            "prepared_development_rows_sha256": manifest["input_hashes"][
                "prepared_development_rows_sha256"
            ],
        },
        "current_code_hashes": _code_hashes(),
        **source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "streamlit": importlib.metadata.version("streamlit"),
        },
        "random_seed": config.random_seed,
        "target_periods": {
            "training": [config.start_date.isoformat(), config.training_end.isoformat()],
            "validation": [config.validation_start.isoformat(), config.validation_end.isoformat()],
            "test": [
                test["target_date"].min().date().isoformat(),
                test["target_date"].max().date().isoformat(),
            ],
        },
        "row_counts": {
            "train": int(len(training)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "excluded_rows_by_split": excluded_rows_by_split,
        "excluded_dates_by_split": excluded_dates_by_split,
        "feature_schema": list(FEATURE_COLUMNS),
        "artifact_schemas": {name: FINAL_RUN_SCHEMA_VERSION for name in payloads},
        "artifact_sha256": artifact_hashes,
        "public_demo_bundle": {
            "status": "not_created",
            "reason": "publication and upload are outside this implementation task",
        },
    }
    if source_experiment is not None:
        run_manifest["source_experiment_id"] = source_experiment["run_id"]
    payloads["run_manifest.json"] = _json_text(run_manifest)

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{safe_id}.", dir=paths.runs_dir))
    try:
        for name, content in payloads.items():
            _atomic_write_text(temporary_dir / name, content)
        if run_dir.exists():
            raise FileExistsError(f"finalized run directory already exists: {run_dir}")
        temporary_dir.rename(run_dir)
    except Exception:
        for item in temporary_dir.iterdir():
            item.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise
    return FinalizedRun(
        run_id=safe_id,
        run_dir=run_dir,
        row_counts=run_manifest["row_counts"],
        selected_model=run_manifest["selected_model_frozen_on_validation"],
    )


def load_finalized_run(run_dir: Path) -> dict[str, Any]:
    """Validate and load only a finalized run directory for report generation."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise DataError(f"run manifest is unreadable: {error}") from error
    manifest = _read_finalized_json_object(manifest_bytes, "run manifest")
    artifact_type = manifest.get("artifact_type")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != FINAL_RUN_SCHEMA_VERSION
        or not isinstance(artifact_type, str)
        or artifact_type
        not in {"sunreserve_finalized_historical_replay", "sunreserve_finalized_experiment"}
        or manifest.get("partition") != "test"
        or manifest.get("test_inspected") is not True
    ):
        raise DataError("run manifest is incompatible or does not record an inspected final test")
    has_excluded_counts = "excluded_rows_by_split" in manifest
    has_excluded_dates = "excluded_dates_by_split" in manifest
    if has_excluded_counts != has_excluded_dates:
        raise DataError("run manifest excluded-row accounting is incomplete")
    legacy_row_counts = manifest.get("row_counts")
    legacy_historical_run = (
        artifact_type == "sunreserve_finalized_historical_replay"
        and not has_excluded_counts
        and not has_excluded_dates
        and "model_bundle" not in manifest
        and isinstance(legacy_row_counts, dict)
        and type(legacy_row_counts.get("test_excluded")) is int
        and legacy_row_counts["test_excluded"] >= 0
    )
    if not legacy_historical_run and not (has_excluded_counts and has_excluded_dates):
        raise DataError("run manifest excluded-row accounting is incomplete")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or _safe_run_id(run_id) != run_dir.name:
        raise DataError("run manifest identifier does not match its finalized directory")
    if artifact_type == "sunreserve_finalized_experiment":
        source_id = manifest.get("source_experiment_id")
        if not isinstance(source_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", source_id
        ):
            raise DataError("finalized experiment run has an invalid source experiment ID")
    expected_names = {
        "predictions.csv",
        "forecast_metrics.json",
        "decision_metrics.csv",
        "decision_predictions.csv",
        "data_quality.json",
        "selection_manifest.json",
        "assumptions.md",
    }
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_names:
        raise DataError("run manifest does not include every required final artifact checksum")
    artifact_schemas = manifest.get("artifact_schemas")
    if (
        not isinstance(artifact_schemas, dict)
        or set(artifact_schemas) != expected_names
        or any(
            type(artifact_schemas[name]) is not int
            or artifact_schemas[name] != FINAL_RUN_SCHEMA_VERSION
            for name in expected_names
        )
    ):
        raise DataError("run manifest artifact schema declarations are incomplete or unsupported")
    artifact_bytes: dict[str, bytes] = {}
    for name in sorted(expected_names):
        path = run_dir / name
        try:
            artifact_bytes[name] = path.read_bytes()
        except OSError as error:
            raise DataError(f"finalized artifact is missing or has changed: {name}") from error
        expected_hash = artifact_hashes[name]
        actual_hash = hashlib.sha256(artifact_bytes[name]).hexdigest()
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise DataError(f"finalized artifact is missing or has changed: {name}")
    if hashlib.sha256(artifact_bytes["selection_manifest.json"]).hexdigest() != manifest.get(
        "selection_manifest_sha256"
    ):
        raise DataError("run selection manifest hash does not match the recorded frozen digest")

    forecast_payload = _read_finalized_json_object(
        artifact_bytes["forecast_metrics.json"], "forecast metrics"
    )
    quality_payload = _read_finalized_json_object(
        artifact_bytes["data_quality.json"], "data quality report"
    )
    selection_payload = _read_finalized_json_object(
        artifact_bytes["selection_manifest.json"], "selection manifest"
    )
    if (
        type(quality_payload.get("schema_version")) is not int
        or quality_payload["schema_version"] != SCHEMA_VERSION
        or type(selection_payload.get("schema_version")) is not int
        or selection_payload["schema_version"] != ARTIFACT_SCHEMA_VERSION
    ):
        raise DataError("finalized data quality or selection manifest schema is unsupported")
    model_identity: dict[str, Any] | None = None
    if "model_bundle" not in manifest:
        if not legacy_historical_run:
            raise DataError("finalized run is missing its frozen model bundle identity")
    else:
        candidate_identity = manifest["model_bundle"]
        expected_identity_fields = {
            "path",
            "model_id",
            "sha256",
            "training_split",
            "training_rows",
            "training_target_period",
        }
        if (
            not isinstance(candidate_identity, dict)
            or set(candidate_identity) != expected_identity_fields
            or not isinstance(candidate_identity.get("path"), str)
            or "\\" in candidate_identity["path"]
            or candidate_identity["path"].startswith("/")
            or any(part in {"", ".", ".."} for part in candidate_identity["path"].split("/"))
            or candidate_identity.get("model_id") != MODEL_ID
            or not isinstance(candidate_identity.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", candidate_identity["sha256"])
            or candidate_identity.get("training_split") != "train"
            or type(candidate_identity.get("training_rows")) is not int
            or candidate_identity["training_rows"] <= 0
        ):
            raise DataError("finalized run has an invalid frozen model bundle identity")
        model_period = candidate_identity.get("training_target_period")
        if (
            not isinstance(model_period, list)
            or len(model_period) != 2
            or any(
                not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
                for value in model_period
            )
        ):
            raise DataError("finalized run has an invalid model training period")
        try:
            normalized_model_period = [
                datetime.strptime(value, "%Y-%m-%d").date().isoformat() for value in model_period
            ]
        except ValueError as error:
            raise DataError(
                "finalized run model training period contains an invalid date"
            ) from error
        if (
            normalized_model_period != model_period
            or normalized_model_period[0] > normalized_model_period[1]
        ):
            raise DataError("finalized run has an invalid model training period")
        model_identity = candidate_identity
        selection_model_identity = selection_payload.get("model_bundle")
        if (
            not isinstance(selection_model_identity, dict)
            or set(selection_model_identity) != expected_identity_fields
            or type(selection_model_identity.get("training_rows")) is not int
            or selection_model_identity != model_identity
        ):
            raise DataError(
                "finalized model bundle identity does not match the frozen selection manifest"
            )

    selected_model = manifest.get("selected_model_frozen_on_validation")
    if not isinstance(selected_model, str) or selected_model not in FINAL_FORECAST_APPROACHES:
        raise DataError("run manifest selected model is unsupported")
    target_periods = manifest.get("target_periods")
    test_period = target_periods.get("test") if isinstance(target_periods, dict) else None
    if (
        not isinstance(test_period, list)
        or len(test_period) != 2
        or any(
            not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            for value in test_period
        )
    ):
        raise DataError("run manifest test target period is invalid")
    try:
        normalized_period = [
            datetime.strptime(value, "%Y-%m-%d").date().isoformat() for value in test_period
        ]
    except ValueError as error:
        raise DataError("run manifest test target period contains an invalid date") from error
    if normalized_period != test_period or normalized_period[0] > normalized_period[1]:
        raise DataError("run manifest test target period is invalid")
    row_counts = manifest.get("row_counts")
    if (
        not isinstance(row_counts, dict)
        or type(row_counts.get("test")) is not int
        or row_counts["test"] <= 0
    ):
        raise DataError("run manifest test row count is invalid")
    if model_identity is not None:
        training_period = target_periods.get("training")
        if (
            type(row_counts.get("train")) is not int
            or row_counts["train"] != model_identity["training_rows"]
            or not isinstance(training_period, list)
            or len(training_period) != 2
            or model_identity["training_target_period"][0] < training_period[0]
            or model_identity["training_target_period"][1] > training_period[1]
        ):
            raise DataError("finalized model bundle identity differs from the training manifest")

    try:
        prediction_frame = pd.read_csv(io.BytesIO(artifact_bytes["predictions.csv"]))
        decision_frame = pd.read_csv(io.BytesIO(artifact_bytes["decision_metrics.csv"]))
        decision_rows = pd.read_csv(io.BytesIO(artifact_bytes["decision_predictions.csv"]))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DataError(f"finalized tabular artifacts are unreadable: {error}") from error
    _require_finalized_columns(prediction_frame, FINAL_PREDICTION_COLUMNS, "predictions.csv")
    _require_finalized_columns(
        decision_frame, FINAL_DECISION_METRIC_COLUMNS, "decision_metrics.csv"
    )
    _require_finalized_columns(
        decision_rows, FINAL_DECISION_PREDICTION_COLUMNS, "decision_predictions.csv"
    )

    for frame, label in (
        (prediction_frame, "predictions.csv"),
        (decision_frame, "decision_metrics.csv"),
        (decision_rows, "decision_predictions.csv"),
    ):
        try:
            versions = pd.to_numeric(frame["artifact_schema_version"], errors="raise").to_numpy(
                dtype=float
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise DataError(f"{label} schema version is unsupported") from error
        if (
            not np.isfinite(versions).all()
            or not np.equal(versions, np.floor(versions)).all()
            or not (versions == FINAL_RUN_SCHEMA_VERSION).all()
        ):
            raise DataError(f"{label} schema version is unsupported")

    if prediction_frame.empty:
        raise DataError("predictions.csv must contain valid final-test target dates")
    prediction_frame["issue_date"] = _finalized_dates(
        prediction_frame["issue_date"], "predictions.csv issue_date"
    )
    prediction_frame["target_date"] = _finalized_dates(
        prediction_frame["target_date"], "predictions.csv target_date"
    )
    if decision_rows.empty:
        raise DataError("decision_predictions.csv must contain final-test rows only")
    decision_rows["target_date"] = _finalized_dates(
        decision_rows["target_date"], "decision_predictions.csv target_date"
    )
    if prediction_frame["split"].isna().any() or set(prediction_frame["split"]) != {"test"}:
        raise DataError("predictions.csv must contain final-test rows only")
    if prediction_frame["selected_model"].isna().any() or set(
        prediction_frame["selected_model"]
    ) != {selected_model}:
        raise DataError("predictions.csv selected model differs from the run manifest")
    if prediction_frame["target_date"].isna().any():
        raise DataError("predictions.csv must contain valid final-test target dates")
    if (
        prediction_frame["target_date"].duplicated().any()
        or not prediction_frame["target_date"].is_monotonic_increasing
    ):
        raise DataError("predictions.csv target dates must be unique and chronological")
    if not (
        prediction_frame["target_date"] == prediction_frame["issue_date"] + pd.Timedelta(days=1)
    ).all():
        raise DataError("predictions.csv issue and target dates must differ by one calendar day")
    if [
        prediction_frame["target_date"].min().date().isoformat(),
        prediction_frame["target_date"].max().date().isoformat(),
    ] != normalized_period:
        raise DataError("predictions.csv target range differs from the run manifest")
    if row_counts["test"] != len(prediction_frame):
        raise DataError("predictions.csv row count differs from the run manifest")

    split_names = {"train", "validation", "test"}
    test_excluded_dates: list[pd.Timestamp] = []
    if not legacy_historical_run:
        excluded_counts = manifest["excluded_rows_by_split"]
        excluded_dates = manifest["excluded_dates_by_split"]
        if (
            not isinstance(excluded_counts, dict)
            or set(excluded_counts) != split_names
            or not isinstance(excluded_dates, dict)
            or set(excluded_dates) != split_names
        ):
            raise DataError("run manifest excluded-row accounting is incomplete")
        for split_name in sorted(split_names):
            records = excluded_dates[split_name]
            count = excluded_counts[split_name]
            if (
                not isinstance(records, list)
                or type(count) is not int
                or count < 0
                or count != len(records)
            ):
                raise DataError(
                    f"run manifest {split_name} exclusion count does not match its records"
                )
            seen_dates: set[pd.Timestamp] = set()
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "issue_date",
                    "target_date",
                    "missing_or_nonfinite_fields",
                }:
                    raise DataError(
                        f"run manifest {split_name} exclusion record has an invalid schema"
                    )
                issue_date, target_date = (
                    _finalized_dates(pd.Series([record[name]]), f"{split_name} exclusion {name}")[0]
                    for name in ("issue_date", "target_date")
                )
                missing_fields = record["missing_or_nonfinite_fields"]
                if (
                    target_date in seen_dates
                    or target_date != issue_date + pd.Timedelta(days=1)
                    or not isinstance(missing_fields, list)
                    or not missing_fields
                    or any(not isinstance(field, str) or not field for field in missing_fields)
                ):
                    raise DataError(f"run manifest {split_name} exclusion record is invalid")
                seen_dates.add(target_date)
                if split_name == "test":
                    test_excluded_dates.append(target_date)
    predicted_dates = pd.DatetimeIndex(prediction_frame["target_date"])
    if set(test_excluded_dates) & set(predicted_dates):
        raise DataError("test exclusion dates overlap eligible forecast targets")
    eligible_calendar = pd.date_range(predicted_dates.min(), predicted_dates.max(), freq="D")
    expected_eligible_dates = eligible_calendar.difference(
        [
            value
            for value in test_excluded_dates
            if predicted_dates.min() <= value <= predicted_dates.max()
        ]
    )
    if not expected_eligible_dates.equals(predicted_dates):
        raise DataError("predictions.csv test target calendar does not reconcile to exclusions")

    prediction_numeric = FINAL_PREDICTION_COLUMNS - {
        "artifact_schema_version",
        "issue_date",
        "target_date",
        "split",
        "selected_model",
    }
    _require_finalized_numeric_columns(
        prediction_frame,
        prediction_numeric,
        "predictions.csv",
        nonnegative=prediction_numeric,
    )
    for column in prediction_numeric:
        prediction_frame[column] = pd.to_numeric(prediction_frame[column], errors="raise")

    _validate_finalized_forecast_metrics(forecast_payload, prediction_frame, selected_model)

    if decision_frame["partition"].isna().any() or set(decision_frame["partition"]) != {"test"}:
        raise DataError("decision_metrics.csv must contain final-test summaries only")
    if decision_rows["split"].isna().any() or set(decision_rows["split"]) != {"test"}:
        raise DataError("decision_predictions.csv must contain final-test rows only")
    for frame, columns, label in (
        (
            decision_frame,
            FINAL_DECISION_METRIC_COLUMNS
            - {"artifact_schema_version", "partition", "scenario_id", "approach", "selected_model"},
            "decision_metrics.csv",
        ),
        (
            decision_rows,
            FINAL_DECISION_PREDICTION_COLUMNS
            - {"artifact_schema_version", "split", "scenario_id", "approach", "target_date"},
            "decision_predictions.csv",
        ),
    ):
        nonnegative = columns - {
            "paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day",
            "paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day",
            "paired_mean_excess_difference_vs_persistence_kwh_day",
            "paired_mean_excess_difference_vs_seasonal_mean_kwh_day",
        }
        missing_by_approach = (
            {"predicted_radiation_kwh_m2_day", "predicted_generation_proxy_kwh_day"}
            if label == "decision_predictions.csv"
            else set()
        )
        _require_finalized_numeric_columns(
            frame,
            columns,
            label,
            nonnegative=nonnegative,
            allow_nan_by_approach=missing_by_approach,
        )
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    for frame, label in (
        (decision_frame, "decision_metrics.csv"),
        (decision_rows, "decision_predictions.csv"),
    ):
        for column in ("scenario_id", "approach"):
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise DataError(f"{label} contains missing scenario/approach keys")
            frame[column] = frame[column].astype(str)
    if decision_frame["selected_model"].isna().any() or set(decision_frame["selected_model"]) != {
        selected_model
    }:
        raise DataError("decision metrics selected model differs from the run manifest")
    if set(decision_frame["approach"]) != FINAL_DECISION_APPROACHES or len(
        decision_frame
    ) != 28 * len(FINAL_DECISION_APPROACHES):
        raise DataError("decision metrics must compare four approaches on all 28 fixed scenarios")
    if (
        decision_frame.duplicated(["scenario_id", "approach"]).any()
        or len(decision_frame["scenario_id"].unique()) != 28
        or set(decision_rows["approach"]) != FINAL_DECISION_APPROACHES
    ):
        raise DataError("decision metrics contain mismatched approach/scenario rows")
    if decision_rows.duplicated(["scenario_id", "approach", "target_date"]).any():
        raise DataError("decision predictions contain duplicate approach/date rows")
    if len(decision_rows) != len(prediction_frame) * 28 * len(FINAL_DECISION_APPROACHES):
        raise DataError("decision predictions do not cover every test date and scenario")
    if set(decision_frame["scenario_id"]) != set(decision_rows["scenario_id"]):
        raise DataError("decision summary and daily scenario sets do not match")

    metric_count_columns = {
        "sample_count",
        "simulated_shortfall_day_count",
        "capacity_limited_shortfall_day_count",
    }
    for column in metric_count_columns:
        values = decision_frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise DataError(f"decision metrics field {column} must contain integer counts")
    if set(decision_frame["sample_count"].astype(int)) != {len(prediction_frame)}:
        raise DataError("decision metrics do not cover all common test dates")
    for column in ("simulated_shortfall_day_count", "capacity_limited_shortfall_day_count"):
        values = decision_frame[column].to_numpy(dtype=int)
        if (values < 0).any() or (values > len(prediction_frame)).any():
            raise DataError(f"decision metrics field {column} is outside the test sample")
    if (decision_frame["simulated_shortfall_day_share"] > 1).any() or (
        decision_frame["simulated_shortfall_day_share"] < 0
    ).any():
        raise DataError("decision shortfall shares must be between zero and one")

    expected_target_dates = pd.DatetimeIndex(prediction_frame["target_date"])
    setting_columns = ["demand_kwh", "capacity_kwh", "buffer_kwh"]
    shared_columns = [*setting_columns, "actual_radiation_kwh_m2_day"]
    for _, group in decision_frame.groupby("scenario_id", sort=False):
        for column in setting_columns:
            if group[column].nunique(dropna=False) != 1:
                raise DataError("decision metrics compare approaches under mismatched settings")
    for _, group in decision_rows.groupby("scenario_id", sort=False):
        unique_dates = pd.DatetimeIndex(group["target_date"].drop_duplicates().sort_values())
        if len(group) != len(prediction_frame) * len(
            FINAL_DECISION_APPROACHES
        ) or not unique_dates.equals(expected_target_dates):
            raise DataError("decision scenarios do not share the forecast target dates")
    for _, group in decision_rows.groupby(["scenario_id", "target_date"], sort=False):
        if set(group["approach"]) != FINAL_DECISION_APPROACHES or any(
            group[column].nunique(dropna=False) != 1 for column in shared_columns
        ):
            raise DataError("decision predictions use mismatched dates or scenario settings")
    actual_by_date = decision_rows.drop_duplicates("target_date").sort_values("target_date")
    if not np.allclose(
        actual_by_date["actual_radiation_kwh_m2_day"].to_numpy(dtype=float),
        prediction_frame["actual_radiation_kwh_m2_day"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise DataError("decision observations differ from the common forecast test series")

    forecasts_by_approach = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }
    forecast_values = prediction_frame.set_index("target_date")
    for approach, column in forecasts_by_approach.items():
        approach_rows = decision_rows.loc[decision_rows["approach"] == approach]
        expected = approach_rows["target_date"].map(forecast_values[column]).to_numpy(dtype=float)
        if not np.allclose(
            approach_rows["predicted_radiation_kwh_m2_day"].to_numpy(dtype=float),
            np.maximum(expected, 0.0),
            rtol=0,
            atol=1e-12,
        ):
            raise DataError("decision predictions differ from the common forecast approaches")

    summary = decision_frame.set_index(["scenario_id", "approach"])
    for key, group in decision_rows.groupby(["scenario_id", "approach"], sort=False):
        row = summary.loc[key]
        shortfall = group["simulated_shortfall_kwh"].to_numpy(dtype=float)
        excess = group["excess_prepared_reserve_kwh"].to_numpy(dtype=float)
        capacity_limited = group["capacity_limited_shortfall_kwh"].to_numpy(dtype=float)
        additional = group["additional_shortfall_above_capacity_floor_kwh"].to_numpy(dtype=float)
        expected_summary = {
            "sample_count": len(group),
            "simulated_shortfall_day_count": int(np.count_nonzero(shortfall > 0)),
            "simulated_shortfall_day_share": float(np.mean(shortfall > 0)),
            "total_simulated_shortfall_kwh": float(shortfall.sum()),
            "mean_simulated_shortfall_kwh_day": float(shortfall.mean()),
            "total_excess_prepared_reserve_kwh": float(excess.sum()),
            "mean_excess_prepared_reserve_kwh_day": float(excess.mean()),
            "capacity_limited_shortfall_day_count": int(np.count_nonzero(capacity_limited > 0)),
            "total_capacity_limited_shortfall_kwh": float(capacity_limited.sum()),
            "mean_capacity_limited_shortfall_kwh_day": float(capacity_limited.mean()),
            "total_additional_shortfall_above_capacity_floor_kwh": float(additional.sum()),
            "mean_additional_shortfall_above_capacity_floor_kwh_day": float(additional.mean()),
        }
        for column, expected in expected_summary.items():
            actual_value = row[column]
            if isinstance(expected, int):
                matches = int(actual_value) == expected
            else:
                matches = math.isclose(float(actual_value), expected, rel_tol=1e-10, abs_tol=1e-10)
            if not matches:
                raise DataError("decision summary metrics do not match daily decision predictions")

    for _, group in decision_frame.groupby("scenario_id", sort=False):
        means = {
            approach: {
                "shortfall": float(row["mean_simulated_shortfall_kwh_day"]),
                "excess": float(row["mean_excess_prepared_reserve_kwh_day"]),
            }
            for approach, row in group.set_index("approach").iterrows()
        }
        for _, row in group.iterrows():
            approach = row["approach"]
            for baseline in ("persistence", "seasonal_mean"):
                expected_shortfall = means[approach]["shortfall"] - means[baseline]["shortfall"]
                expected_excess = means[approach]["excess"] - means[baseline]["excess"]
                for column, expected in (
                    (
                        f"paired_mean_simulated_shortfall_difference_vs_{baseline}_kwh_day",
                        expected_shortfall,
                    ),
                    (
                        f"paired_mean_excess_difference_vs_{baseline}_kwh_day",
                        expected_excess,
                    ),
                ):
                    if not math.isclose(float(row[column]), expected, rel_tol=1e-10, abs_tol=1e-10):
                        raise DataError(
                            "decision paired differences do not match scenario summaries"
                        )

    try:
        assumptions = artifact_bytes["assumptions.md"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise DataError(f"finalized assumptions are unreadable: {error}") from error
    return {
        "run_manifest": manifest,
        "forecast_metrics": forecast_payload,
        "predictions": prediction_frame,
        "decision_metrics": decision_frame,
        "decision_predictions": decision_rows,
        "assumptions": assumptions,
    }


def render_final_report(run_dir: Path) -> Path:
    """Generate a deterministic report from artifacts inside one finalized run only."""
    finalized = load_finalized_run(run_dir)
    run_manifest = finalized["run_manifest"]
    forecast_payload = finalized["forecast_metrics"]
    predictions = finalized["predictions"]
    decision_metrics = finalized["decision_metrics"]
    selected_model = run_manifest["selected_model_frozen_on_validation"]
    forecast_results = forecast_payload["metrics"]
    model_columns = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }

    lines = [
        "# SunReserve final historical replay",
        "",
        "This report evaluates the frozen validation selection on the explicitly finalized "
        "held-out test period. It is a historical research replay, not a "
        "validated operational controller.",
        "",
        f"- Run: `{run_manifest['run_id']}`",
        f"- Test period: {run_manifest['target_periods']['test'][0]} through "
        f"{run_manifest['target_periods']['test'][1]} "
        f"({run_manifest['row_counts']['test']} eligible targets)",
        f"- Model selected using validation only: **{selected_model}**",
        f"- Test inspection recorded: {run_manifest['test_inspected_utc']}",
        "- Identity: `uncommitted`; code identity is recorded as source file SHA-256 values.",
        "",
        "## Forecast comparison",
        "",
        "Solar-radiation units are kWh/m²/day. Model selection was frozen before these "
        "held-out metrics were calculated. Persistence and the training-only seasonal "
        "mean remain included.",
        "",
        "| Approach | MAE | RMSE | Bias (prediction − target) | Low-solar MAE | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for approach in SELECTION_TIE_ORDER:
        metric = forecast_results[approach]
        low_mae = metric["low_solar"]["mae"]
        lines.append(
            f"| {approach} | {metric['mae']:.4f} | {metric['rmse']:.4f} | "
            f"{metric['bias_prediction_minus_actual']:.4f} | "
            f"{'n/a' if low_mae is None else f'{low_mae:.4f}'} | {metric['sample_count']} |"
        )
    lines.extend(["", "## Seasonal behavior", "", "Seasonal MAE by approach (kWh/m²/day):", ""])
    all_seasons = sorted(
        {season for metric in forecast_results.values() for season in metric["seasonal_mae"]}
    )
    lines.append("| Approach | " + " | ".join(all_seasons) + " |")
    lines.append("|---|" + "---:|" * len(all_seasons))
    for approach in SELECTION_TIE_ORDER:
        seasonal = forecast_results[approach]["seasonal_mae"]
        values = [
            "n/a" if season not in seasonal else f"{seasonal[season]:.4f}" for season in all_seasons
        ]
        lines.append(f"| {approach} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Reserve comparison",
            "",
            "All reserve and electricity values are hypothetical simulated kWh/day. "
            "Shortfall and excess reserve are separate quantities; no monetary or emissions "
            "savings are inferred.",
            "",
            "### Primary hypothetical scenario",
            "",
            "| Approach | Total simulated shortfall (kWh) | Shortfall days | "
            "Total excess prepared reserve (kWh) | Capacity-limited floor (kWh) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    primary = decision_metrics.loc[decision_metrics["scenario_id"] == "primary"]
    for approach in (*SELECTION_TIE_ORDER, "always_prepare_backup"):
        row = primary.loc[primary["approach"] == approach].iloc[0]
        lines.append(
            f"| {approach} | {row['total_simulated_shortfall_kwh']:.2f} | "
            f"{int(row['simulated_shortfall_day_count'])} | "
            f"{row['total_excess_prepared_reserve_kwh']:.2f} | "
            f"{row['total_capacity_limited_shortfall_kwh']:.2f} |"
        )
    grid = decision_metrics.loc[decision_metrics["scenario_id"] != "primary"]
    lines.extend(
        [
            "",
            "### Fixed sensitivity grid",
            "",
            f"The report includes {grid['scenario_id'].nunique()} predeclared "
            "demand/capacity/buffer "
            "settings for every forecast approach and the always-prepare reference.",
            "",
            "| Approach | Mean daily shortfall range (kWh/day) | "
            "Mean excess reserve range (kWh/day) |",
            "|---|---:|---:|",
        ]
    )
    for approach in (*SELECTION_TIE_ORDER, "always_prepare_backup"):
        subset = grid.loc[grid["approach"] == approach]
        shortfall_min = subset["mean_simulated_shortfall_kwh_day"].min()
        shortfall_max = subset["mean_simulated_shortfall_kwh_day"].max()
        excess_min = subset["mean_excess_prepared_reserve_kwh_day"].min()
        excess_max = subset["mean_excess_prepared_reserve_kwh_day"].max()
        lines.append(
            f"| {approach} | {shortfall_min:.4f}–{shortfall_max:.4f} | "
            f"{excess_min:.4f}–{excess_max:.4f} |"
        )

    actual = predictions["actual_radiation_kwh_m2_day"].to_numpy(dtype=float)
    selected_prediction = predictions[model_columns[selected_model]].to_numpy(dtype=float)
    prediction_errors = selected_prediction - actual
    error_frame = predictions[["target_date"]].copy()
    error_frame["prediction_error"] = prediction_errors
    largest_over = error_frame.loc[error_frame["prediction_error"] > 0].nlargest(
        1, "prediction_error"
    )
    largest_under = error_frame.loc[error_frame["prediction_error"] < 0].nsmallest(
        1, "prediction_error"
    )
    lines.extend(
        [
            "",
            "## Unfavorable forecast examples",
            "",
            f"For the validation-selected `{selected_model}` approach, examples are selected by "
            "the largest positive and largest negative test prediction errors; "
            "this rule does not filter for favorable days.",
            "",
            "| Example | Target date | Prediction minus target (kWh/m²/day) |",
            "|---|---|---:|",
        ]
    )
    for label, example in (
        ("Largest overprediction", largest_over),
        ("Largest underprediction", largest_under),
    ):
        if example.empty:
            lines.append(f"| {label} | n/a | n/a |")
        else:
            row = example.iloc[0]
            lines.append(
                f"| {label} | {row['target_date'].date().isoformat()} | "
                f"{row['prediction_error']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- NASA POWER supplies model/satellite-derived daily solar radiation; "
            "it is not measured electricity generation.",
            "- Demand, array area, conversion efficiency, derating, reserve capacity, and "
            "buffer are hypothetical assumptions.",
            "- The generation proxy is not a validated PV yield model; the reserve "
            "calculation has no inter-day battery state, costs, outages, or dispatch constraints.",
            "- Forecast-error metrics and reserve metrics answer different questions. "
            "The replay is not a causal savings or emissions analysis.",
            "- Historical data availability does not establish operational forecast "
            "availability. No real facility deployment is demonstrated.",
            "",
            "The full run settings are recorded in `selection_manifest.json`, "
            "`run_manifest.json`, and `assumptions.md`.",
            "",
        ]
    )
    report_text = "\n".join(lines)
    report_path = Path(run_dir) / "report.md"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != report_text:
            raise FileExistsError(
                f"report exists with different contents and will not be overwritten: {report_path}"
            )
    else:
        _atomic_write_text(report_path, report_text)
    return report_path
