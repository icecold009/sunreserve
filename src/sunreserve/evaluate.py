"""Validation-only model comparison, metrics, and reproducible selection artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import DataError, ProjectPaths, request_url
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
    build_supervised_rows,
    split_supervised_rows,
)
from sunreserve.inference import write_model_bundle
from sunreserve.train import baseline_predictions, fit_ridge, predict_ridge

ARTIFACT_SCHEMA_VERSION = 2
FINAL_RUN_SCHEMA_VERSION = 1
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


def load_development_daily(paths: ProjectPaths, config: ExperimentConfig) -> pd.DataFrame:
    """Load only data through the end of validation; never read final-test rows."""
    if not paths.daily_path.is_file():
        raise DataError(
            f"prepared daily data is missing at {paths.daily_path}; run fetch and prepare first"
        )
    for path, label in (
        (paths.quality_path, "quality report"),
        (paths.sidecar_path, "raw provenance sidecar"),
    ):
        if not path.is_file():
            raise DataError(f"{label} is missing at {path}; run fetch and prepare first")
    try:
        quality = json.loads(paths.quality_path.read_text(encoding="utf-8"))
        provenance = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataError(f"cannot read prepared data provenance: {error}") from error
    if quality.get("requested_start") != config.start_date.isoformat():
        raise DataError("prepared data start date does not match active configuration")
    if quality.get("requested_end") != config.end_date.isoformat():
        raise DataError("prepared data end date does not match active configuration")
    if provenance.get("request_url") != request_url(config):
        raise DataError("raw data provenance does not match active configuration")
    if provenance.get("sha256") != quality.get("raw_sha256"):
        raise DataError("prepared quality report does not match raw response checksum")
    if not quality.get("prepared_daily_csv_sha256"):
        raise DataError("prepared daily checksum is missing; run prepare before validation")

    development_days = (config.validation_end - config.start_date).days + 1
    if development_days <= 0:
        raise DataError("validation end date must follow the experiment start date")
    try:
        data = pd.read_csv(paths.daily_path, nrows=development_days, parse_dates=["date"])
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataError(f"cannot read prepared development data: {error}") from error
    expected = pd.date_range(config.start_date, config.validation_end, freq="D")
    if len(data) != len(expected):
        raise DataError(
            f"prepared development data has {len(data)} rows; expected {len(expected)} "
            "through the validation boundary"
        )
    dates = pd.DatetimeIndex(data["date"])
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


def run_validation(paths: ProjectPaths, config: ExperimentConfig) -> ValidationSelection:
    """Run cached validation-only selection and write versioned development artifacts."""
    daily = load_development_daily(paths, config)
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
    model_bundle, model_bundle_sha256 = write_model_bundle(
        partitions["train"], config, paths.model_path
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
        "code_revision": "uncommitted",
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
            "path": paths.model_path.relative_to(paths.root).as_posix(),
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
    _atomic_write_text(paths.validation_dir / "validation_predictions.csv", predictions_content)
    _atomic_write_text(paths.validation_dir / "validation_metrics.json", metrics_content)
    _atomic_write_text(paths.validation_dir / "selection_manifest.json", manifest_content)
    _atomic_write_text(
        paths.validation_dir / "decision_predictions.csv", decision_predictions_content
    )
    _atomic_write_text(paths.validation_dir / "decision_metrics.csv", decision_metrics_content)
    return selection


@dataclass(frozen=True)
class FinalizedRun:
    """Identity and test-row counts for one immutable finalized experiment run."""

    run_id: str
    run_dir: Path
    row_counts: dict[str, int]
    selected_model: str


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
) -> FinalizedRun:
    """Evaluate the held-out test only after an explicitly frozen, verified manifest."""
    safe_id = _safe_run_id(run_id)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = paths.runs_dir / safe_id
    if run_dir.exists():
        raise FileExistsError(
            f"finalized run directory already exists and will not be overwritten: {run_dir}"
        )

    manifest, config, quality, _development = _validate_frozen_development(
        paths,
        expected_manifest_sha256=manifest_sha256,
        expected_config=expected_config,
    )
    daily = _load_full_daily(paths, config)
    rows, split_audit = build_supervised_rows(daily, config)
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

    predictions = baseline_predictions(training, test, allow_final_test=True)
    ridge = fit_ridge(training, config)
    predictions["ridge_prediction_kwh_m2_day"] = predict_ridge(ridge, test).to_numpy()
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

    selection_manifest_bytes = (paths.validation_dir / "selection_manifest.json").read_bytes()
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
        "artifact_type": "sunreserve_finalized_historical_replay",
        "run_id": safe_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "partition": "test",
        "test_inspected": True,
        "test_inspected_utc": datetime.now(UTC).isoformat(),
        "selected_model_frozen_on_validation": manifest["model_selection"]["selected_model"],
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
        "code_revision": "uncommitted",
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
            "test_excluded": int(split_audit["excluded_rows"]),
        },
        "feature_schema": list(FEATURE_COLUMNS),
        "artifact_schemas": {name: FINAL_RUN_SCHEMA_VERSION for name in payloads},
        "artifact_sha256": artifact_hashes,
        "public_demo_bundle": {
            "status": "not_created",
            "reason": "publication and upload are outside this implementation task",
        },
    }
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
    manifest = _read_json_object(manifest_path, "run manifest")
    if (
        manifest.get("schema_version") != FINAL_RUN_SCHEMA_VERSION
        or manifest.get("artifact_type") != "sunreserve_finalized_historical_replay"
        or manifest.get("partition") != "test"
        or manifest.get("test_inspected") is not True
    ):
        raise DataError("run manifest is incompatible or does not record an inspected final test")
    if _safe_run_id(manifest.get("run_id")) != run_dir.name:
        raise DataError("run manifest identifier does not match its finalized directory")
    expected_names = {
        "predictions.csv",
        "forecast_metrics.json",
        "decision_metrics.csv",
        "decision_predictions.csv",
        "data_quality.json",
        "selection_manifest.json",
        "assumptions.md",
    }
    if set(manifest.get("artifact_sha256", {})) != expected_names:
        raise DataError("run manifest does not include every required final artifact checksum")
    expected_schemas = {name: FINAL_RUN_SCHEMA_VERSION for name in expected_names}
    if manifest.get("artifact_schemas") != expected_schemas:
        raise DataError("run manifest artifact schema declarations are incomplete or unsupported")
    for name, expected_hash in manifest["artifact_sha256"].items():
        path = run_dir / name
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise DataError(f"finalized artifact is missing or has changed: {name}")
    if _sha256_file(run_dir / "selection_manifest.json") != manifest.get(
        "selection_manifest_sha256"
    ):
        raise DataError("run selection manifest hash does not match the recorded frozen digest")

    forecast_payload = _read_json_object(run_dir / "forecast_metrics.json", "forecast metrics")
    if forecast_payload.get("schema_version") != FINAL_RUN_SCHEMA_VERSION:
        raise DataError("forecast metrics schema version is unsupported")
    if (
        forecast_payload.get("partition") != "test"
        or forecast_payload.get("target_units") != "kWh/m^2/day"
    ):
        raise DataError("forecast metrics must describe the test partition in radiation units")
    try:
        prediction_frame = pd.read_csv(
            run_dir / "predictions.csv", parse_dates=["issue_date", "target_date"]
        )
        decision_frame = pd.read_csv(run_dir / "decision_metrics.csv")
        decision_rows = pd.read_csv(
            run_dir / "decision_predictions.csv", parse_dates=["target_date"]
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise DataError(f"finalized tabular artifacts are unreadable: {error}") from error
    if "artifact_schema_version" not in prediction_frame or set(
        prediction_frame["artifact_schema_version"]
    ) != {FINAL_RUN_SCHEMA_VERSION}:
        raise DataError("predictions.csv schema version is unsupported")
    if set(prediction_frame.get("split", [])) != {"test"}:
        raise DataError("predictions.csv must contain final-test rows only")
    expected_predictions = {
        "issue_date",
        "target_date",
        "actual_radiation_kwh_m2_day",
        "persistence_prediction_kwh_m2_day",
        "seasonal_mean_prediction_kwh_m2_day",
        "ridge_prediction_kwh_m2_day",
    }
    if not expected_predictions <= set(prediction_frame.columns):
        raise DataError("predictions.csv is missing required dated radiation columns")
    if prediction_frame.empty or prediction_frame["target_date"].isna().any():
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
    target_period = manifest.get("target_periods", {}).get("test", [])
    if (
        len(target_period) != 2
        or [
            prediction_frame["target_date"].min().date().isoformat(),
            prediction_frame["target_date"].max().date().isoformat(),
        ]
        != target_period
    ):
        raise DataError("predictions.csv target range differs from the run manifest")
    row_counts = manifest.get("row_counts", {})
    if row_counts.get("test") != len(prediction_frame):
        raise DataError("predictions.csv row count differs from the run manifest")
    prediction_values = prediction_frame[
        sorted(expected_predictions - {"issue_date", "target_date"})
    ].to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise DataError("predictions.csv contains non-finite radiation values")
    if set(forecast_payload.get("metrics", {})) != {
        "persistence",
        "seasonal_mean",
        "ridge",
    } or any(
        metrics.get("sample_count") != len(prediction_frame)
        for metrics in forecast_payload["metrics"].values()
    ):
        raise DataError("forecast metrics do not cover all approaches and common test dates")
    if "artifact_schema_version" not in decision_frame or set(
        decision_frame["artifact_schema_version"]
    ) != {FINAL_RUN_SCHEMA_VERSION}:
        raise DataError("decision_metrics.csv schema version is unsupported")
    if set(decision_frame.get("partition", [])) != {"test"}:
        raise DataError("decision_metrics.csv must contain final-test summaries only")
    if len(decision_frame) != 28 * 4 or set(decision_frame.get("approach", [])) != {
        "persistence",
        "seasonal_mean",
        "ridge",
        "always_prepare_backup",
    }:
        raise DataError("decision metrics must compare four approaches on all 28 fixed scenarios")
    if decision_frame.duplicated(["scenario_id", "approach"]).any() or not all(
        set(group["approach"]) == {"persistence", "seasonal_mean", "ridge", "always_prepare_backup"}
        for _, group in decision_frame.groupby("scenario_id")
    ):
        raise DataError("decision metrics contain mismatched approach/scenario rows")
    setting_columns = ["demand_kwh", "capacity_kwh", "buffer_kwh"]
    if not set(setting_columns) <= set(decision_frame.columns):
        raise DataError("decision metrics do not record their hypothetical scenario settings")
    for _, group in decision_frame.groupby("scenario_id"):
        for column in setting_columns:
            if group[column].nunique(dropna=False) != 1:
                raise DataError("decision metrics compare approaches under mismatched settings")
    if "artifact_schema_version" not in decision_rows or set(
        decision_rows["artifact_schema_version"]
    ) != {FINAL_RUN_SCHEMA_VERSION}:
        raise DataError("decision_predictions.csv schema version is unsupported")
    if set(decision_rows.get("split", [])) != {"test"}:
        raise DataError("decision_predictions.csv must contain final-test rows only")
    if decision_rows.duplicated(["scenario_id", "approach", "target_date"]).any():
        raise DataError("decision predictions contain duplicate approach/date rows")
    if len(decision_rows) != len(prediction_frame) * 28 * 4:
        raise DataError("decision predictions do not cover every test date and scenario")
    if set(decision_rows.get("approach", [])) != {
        "persistence",
        "seasonal_mean",
        "ridge",
        "always_prepare_backup",
    }:
        raise DataError("decision predictions are missing one or more approaches")
    shared_columns = [*setting_columns, "actual_radiation_kwh_m2_day"]
    expected_target_dates = pd.DatetimeIndex(prediction_frame["target_date"])
    for _, group in decision_rows.groupby("scenario_id"):
        if len(group) != len(prediction_frame) * 4 or not pd.DatetimeIndex(
            group["target_date"].drop_duplicates()
        ).equals(expected_target_dates):
            raise DataError("decision scenarios do not share the forecast target dates")
    for _, group in decision_rows.groupby(["scenario_id", "target_date"]):
        if set(group["approach"]) != {
            "persistence",
            "seasonal_mean",
            "ridge",
            "always_prepare_backup",
        } or any(group[column].nunique(dropna=False) != 1 for column in shared_columns):
            raise DataError("decision predictions use mismatched dates or scenario settings")
    shared_actuals = decision_rows.groupby("target_date")["actual_radiation_kwh_m2_day"].nunique(
        dropna=False
    )
    if (shared_actuals != 1).any():
        raise DataError("decision approaches do not share one observed target per date")
    actual_by_date = decision_rows.drop_duplicates("target_date").sort_values("target_date")
    if not np.allclose(
        actual_by_date["actual_radiation_kwh_m2_day"].to_numpy(dtype=float),
        prediction_frame["actual_radiation_kwh_m2_day"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise DataError("decision observations differ from the common forecast test series")
    return {
        "run_manifest": manifest,
        "forecast_metrics": forecast_payload,
        "predictions": prediction_frame,
        "decision_metrics": decision_frame,
        "decision_predictions": decision_rows,
        "assumptions": (run_dir / "assumptions.md").read_text(encoding="utf-8"),
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
