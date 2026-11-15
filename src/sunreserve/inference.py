"""Portable, validated Ridge inference for a recent daily observation window."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.features import FEATURE_COLUMNS, RADIATION, build_next_day_features
from sunreserve.train import fit_ridge, fit_seasonal_mean

MODEL_SCHEMA_VERSION = 1
MODEL_ID = "sunreserve.ridge.train-only.v1"
OBSERVATION_FIELDS = ("date", RADIATION, "T2M", "RH2M")


class ModelArtifactError(ValueError):
    """Raised when a saved model bundle is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class RidgeModelBundle:
    """Numeric parts of a fitted sklearn Ridge pipeline and seasonal baseline."""

    model_id: str
    feature_columns: tuple[str, ...]
    imputer_statistics: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    monthly_means: dict[int, float]
    training_global_mean: float
    training_rows: int
    training_start: date
    training_end: date
    artifact_sha256: str | None = None


@dataclass(frozen=True)
class PredictionResult:
    """One next-calendar-day radiation prediction and its simple baselines."""

    issue_date: date
    target_date: date
    radiation_kwh_m2_day: float
    persistence_kwh_m2_day: float
    seasonal_mean_kwh_m2_day: float
    model_id: str
    features: dict[str, float]


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelArtifactError(f"model JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def model_bundle_payload(training_rows: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Fit model and baseline state using only rows explicitly labeled train."""
    if training_rows.empty or set(training_rows.get("split", pd.Series(dtype=str))) != {"train"}:
        raise ValueError("model bundle can only be built from a nonempty training partition")
    targets = pd.to_datetime(training_rows["target_date"], errors="raise")
    if targets.max().date() > config.training_end:
        raise ValueError("model bundle training rows extend beyond configured training_end")
    if targets.min() < pd.Timestamp(config.start_date) or targets.max() > pd.Timestamp(
        config.training_end
    ):
        raise ValueError("model bundle target dates fall outside the configured training period")

    model = fit_ridge(training_rows, config)
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    seasonal = fit_seasonal_mean(training_rows)
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_type": "sunreserve_fitted_ridge_model",
        "model_id": MODEL_ID,
        "target": "next-calendar-day NASA POWER daily solar radiation",
        "target_units": "kWh/m^2/day",
        "feature_columns": list(FEATURE_COLUMNS),
        "training": {
            "split": "train",
            "row_count": int(len(training_rows)),
            "target_start": targets.min().date().isoformat(),
            "target_end": targets.max().date().isoformat(),
            "training_end_limit": config.training_end.isoformat(),
            "random_seed": config.random_seed,
            "ridge_alpha": config.ridge_alpha,
        },
        "preprocessing": {
            "imputation": "median fitted on training rows",
            "imputer_statistics": [float(value) for value in imputer.statistics_],
            "scaling": "standard scaling fitted on training rows",
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [float(value) for value in scaler.scale_],
        },
        "ridge": {
            "coefficients": [float(value) for value in ridge.coef_],
            "intercept": float(ridge.intercept_),
            "prediction_clip": "max(0, prediction)",
        },
        "seasonal_mean_baseline": {
            "monthly_means": {
                str(key): float(value) for key, value in seasonal.monthly_means.items()
            },
            "training_global_mean": seasonal.training_global_mean,
        },
        "availability_note": (
            "This model uses historical observation dates; source release timestamps were not "
            "verified, so this artifact does not establish operational day-ahead availability."
        ),
    }


def write_model_bundle(
    training_rows: pd.DataFrame,
    config: ExperimentConfig,
    path: Path,
) -> tuple[RidgeModelBundle, str]:
    """Fit a train-only bundle, atomically serialize it, and return its SHA-256."""
    payload = model_bundle_payload(training_rows, config)
    content = _canonical_json(payload)
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
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return load_model_bundle(path), digest


def _finite_array(value: Any, name: str, expected_length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ModelArtifactError(f"model {name} must be a numeric array") from error
    if array.shape != (expected_length,) or not np.isfinite(array).all():
        raise ModelArtifactError(f"model {name} must contain {expected_length} finite values")
    return array


def _parse_model(payload: dict[str, Any], digest: str) -> RidgeModelBundle:
    if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ModelArtifactError("model schema version is unsupported")
    if payload.get("artifact_type") != "sunreserve_fitted_ridge_model":
        raise ModelArtifactError("file is not a SunReserve fitted Ridge model")
    if payload.get("model_id") != MODEL_ID:
        raise ModelArtifactError("model identity is unsupported")
    feature_columns = payload.get("feature_columns")
    if not isinstance(feature_columns, list) or not all(
        isinstance(value, str) for value in feature_columns
    ):
        raise ModelArtifactError("model feature_columns must be a list of strings")
    if tuple(feature_columns) != FEATURE_COLUMNS:
        raise ModelArtifactError(
            "model feature schema does not match the current feature definitions"
        )
    count = len(FEATURE_COLUMNS)
    preprocessing = payload.get("preprocessing")
    ridge = payload.get("ridge")
    training = payload.get("training")
    baseline = payload.get("seasonal_mean_baseline")
    if not all(isinstance(value, dict) for value in (preprocessing, ridge, training, baseline)):
        raise ModelArtifactError(
            "model bundle is missing preprocessing, training, Ridge, or baseline data"
        )
    if preprocessing.get("imputation") != "median fitted on training rows":
        raise ModelArtifactError("model imputation contract is unsupported")
    if preprocessing.get("scaling") != "standard scaling fitted on training rows":
        raise ModelArtifactError("model scaling contract is unsupported")
    medians = _finite_array(preprocessing.get("imputer_statistics"), "imputer_statistics", count)
    means = _finite_array(preprocessing.get("scaler_mean"), "scaler_mean", count)
    scales = _finite_array(preprocessing.get("scaler_scale"), "scaler_scale", count)
    coefficients = _finite_array(ridge.get("coefficients"), "coefficients", count)
    if (scales <= 0).any():
        raise ModelArtifactError("model scaler scales must be positive")
    try:
        intercept = float(ridge["intercept"])
        training_rows = int(training["row_count"])
        training_start = date.fromisoformat(training["target_start"])
        training_end = date.fromisoformat(training["target_end"])
        training_end_limit = date.fromisoformat(training["training_end_limit"])
        seed = int(training["random_seed"])
        alpha = float(training["ridge_alpha"])
        monthly = {int(key): float(value) for key, value in baseline["monthly_means"].items()}
        global_mean = float(baseline["training_global_mean"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelArtifactError(f"model metadata is invalid: {error}") from error
    numeric_metadata = [intercept, global_mean, alpha, *monthly.values()]
    if not all(math.isfinite(value) for value in numeric_metadata):
        raise ModelArtifactError("model metadata must contain only finite numeric values")
    if training_rows <= 0 or seed < 0 or alpha <= 0:
        raise ModelArtifactError("model training metadata is out of range")
    if training_start > training_end or training_end > training_end_limit:
        raise ModelArtifactError("model training dates exceed their declared training-only period")
    if not monthly or any(month < 1 or month > 12 for month in monthly):
        raise ModelArtifactError("model seasonal baseline contains invalid month keys")
    return RidgeModelBundle(
        model_id=MODEL_ID,
        feature_columns=FEATURE_COLUMNS,
        imputer_statistics=medians,
        scaler_mean=means,
        scaler_scale=scales,
        coefficients=coefficients,
        intercept=intercept,
        monthly_means=monthly,
        training_global_mean=global_mean,
        training_rows=training_rows,
        training_start=training_start,
        training_end=training_end,
        artifact_sha256=digest,
    )


def load_model_bundle(path: Path, *, expected_sha256: str | None = None) -> RidgeModelBundle:
    """Load a numeric model bundle without executing serialized Python objects."""
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelArtifactError(f"trained model is missing or unreadable: {path}") from error
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ModelArtifactError("trained model SHA-256 does not match its selection manifest")
    try:
        payload = json.loads(content, object_pairs_hook=_duplicate_safe_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError(f"trained model JSON is corrupt: {error}") from error
    if not isinstance(payload, dict):
        raise ModelArtifactError("trained model JSON must contain an object")
    try:
        return _parse_model(payload, digest)
    except ModelArtifactError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as error:
        raise ModelArtifactError(f"trained model JSON has invalid field types: {error}") from error


def load_project_model(root: Path) -> tuple[RidgeModelBundle, dict[str, Any]]:
    """Load the default model only when it matches the validation manifest checksum."""
    project_root = root.resolve()
    manifest_path = project_root / "artifacts" / "validation" / "selection_manifest.json"
    model_path = project_root / "models" / "ridge_model.json"
    try:
        manifest_payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
        )
    except OSError as error:
        raise ModelArtifactError(
            f"validation selection manifest is missing or unreadable: {manifest_path}"
        ) from error
    except (json.JSONDecodeError, ModelArtifactError) as error:
        raise ModelArtifactError(f"validation selection manifest is corrupt: {error}") from error
    if not isinstance(manifest_payload, dict):
        raise ModelArtifactError("validation selection manifest must contain an object")
    metadata = manifest_payload.get("model_bundle")
    if not isinstance(metadata, dict):
        raise ModelArtifactError("validation manifest does not identify a trained model bundle")
    if metadata.get("path") != "models/ridge_model.json":
        raise ModelArtifactError("validation manifest points to an unsupported model path")
    expected_sha256 = metadata.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ModelArtifactError("validation manifest has no valid model checksum")
    if manifest_payload.get("status", "").split(";")[0] != "validation_only":
        raise ModelArtifactError("model-selection manifest is not validation-only")
    bundle = load_model_bundle(model_path, expected_sha256=expected_sha256)
    if (
        metadata.get("model_id") != bundle.model_id
        or metadata.get("training_split") != "train"
        or metadata.get("training_rows") != bundle.training_rows
        or metadata.get("training_target_period")
        != [bundle.training_start.isoformat(), bundle.training_end.isoformat()]
    ):
        raise ModelArtifactError("trained model metadata does not match the selection manifest")
    decision_selection = manifest_payload.get("decision_selection", {})
    if not isinstance(decision_selection, dict):
        raise ModelArtifactError("validation manifest has an invalid decision selection")
    return bundle, decision_selection


def observation_frame_from_records(records: Any) -> pd.DataFrame:
    """Convert the CLI/UI record shape to a strict chronological daily frame."""
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError("observations must be a list of exactly seven daily records")
    parsed: list[dict[str, Any]] = []
    dates: list[pd.Timestamp] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"observation {index + 1} must be a JSON object")
        missing = set(OBSERVATION_FIELDS) - set(record)
        extra = set(record) - set(OBSERVATION_FIELDS)
        if missing:
            raise ValueError(
                f"observation {index + 1} is missing fields: {', '.join(sorted(missing))}"
            )
        if extra:
            raise ValueError(
                f"observation {index + 1} has unsupported fields: {', '.join(sorted(extra))}"
            )
        raw_date = record["date"]
        if not isinstance(raw_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            raise ValueError(f"observation {index + 1} date must use ISO YYYY-MM-DD format")
        try:
            dates.append(pd.Timestamp(date.fromisoformat(raw_date)))
        except ValueError as error:
            raise ValueError(
                f"observation {index + 1} date must use ISO YYYY-MM-DD format"
            ) from error
        numeric_fields: dict[str, float] = {}
        for key in (RADIATION, "T2M", "RH2M"):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"observation {index + 1} field {key} must be numeric")
            numeric_fields[key] = float(value)
        parsed.append(numeric_fields)
    frame = pd.DataFrame(parsed, index=pd.DatetimeIndex(dates))
    return frame


def predict_next_day(
    bundle: RidgeModelBundle,
    observations: pd.DataFrame,
) -> PredictionResult:
    """Run fresh fitted Ridge inference plus persistence and seasonal baselines."""
    feature_row = build_next_day_features(observations)
    values = feature_row.loc[:, bundle.feature_columns].to_numpy(dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("inference features must be finite")
    imputed = np.where(np.isfinite(values), values, bundle.imputer_statistics)
    scaled = (imputed - bundle.scaler_mean) / bundle.scaler_scale
    prediction = max(0.0, float(np.dot(scaled, bundle.coefficients) + bundle.intercept))
    latest_radiation = float(observations[RADIATION].iloc[-1])
    target_date = pd.Timestamp(feature_row.loc[0, "target_date"])
    month = target_date.month
    seasonal = max(0.0, bundle.monthly_means.get(month, bundle.training_global_mean))
    return PredictionResult(
        issue_date=pd.Timestamp(feature_row.loc[0, "issue_date"]).date(),
        target_date=target_date.date(),
        radiation_kwh_m2_day=prediction,
        persistence_kwh_m2_day=max(0.0, latest_radiation),
        seasonal_mean_kwh_m2_day=seasonal,
        model_id=bundle.model_id,
        features={column: float(feature_row.loc[0, column]) for column in bundle.feature_columns},
    )


def load_observation_file(path: Path) -> pd.DataFrame:
    """Load the documented CLI observation JSON and validate it as an inference window."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_duplicate_safe_object
        )
    except OSError as error:
        raise ValueError(f"cannot read observation file {path}: {error}") from error
    except (json.JSONDecodeError, ModelArtifactError) as error:
        raise ValueError(f"observation JSON is invalid: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"observations"}:
        raise ValueError('observation JSON must contain exactly one key: "observations"')
    frame = observation_frame_from_records(payload["observations"])
    build_next_day_features(frame)
    return frame
