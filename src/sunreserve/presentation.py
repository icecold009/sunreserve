"""Offline artifact loading and integrity checks for the Streamlit demonstration."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import ProjectPaths, request_url
from sunreserve.decision import (
    APPROACHES,
    BUFFER_FRACTIONS,
    CAPACITY_LEVELS_KWH,
    DEMAND_LEVELS_KWH,
    PREDICTION_COLUMNS,
    RADIATION_COLUMN,
)
from sunreserve.evaluate import ARTIFACT_SCHEMA_VERSION, PREDICTION_CLIP_RULE, forecast_metrics
from sunreserve.features import TARGET_COLUMN

FORECAST_COLUMNS = {
    "issue_date",
    "target_date",
    RADIATION_COLUMN,
    *PREDICTION_COLUMNS.values(),
}
DECISION_COLUMNS = {
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
DECISION_METRIC_COLUMNS = {
    "scenario_id",
    "approach",
    "selected_model",
    "sample_count",
    "total_simulated_shortfall_kwh",
    "mean_simulated_shortfall_kwh_day",
    "simulated_shortfall_day_count",
    "simulated_shortfall_day_share",
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

CHECKSUM_BOUND_OUTPUTS = {
    "validation_predictions.csv",
    "validation_metrics.json",
    "decision_predictions.csv",
    "decision_metrics.csv",
}
MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "configuration",
    "configuration_sha256",
    "input_hashes",
    "validation_target_period",
    "target",
    "target_units",
    "model_selection",
    "decision_simulation",
    "development_artifact_hashes",
}
_SAFE_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
INPUT_HASH_FIELDS = {
    "raw_response_sha256",
    "quality_report_sha256",
    "prepared_daily_csv_sha256",
    "prepared_development_rows_sha256",
}
FORECAST_METRIC_FIELDS = {
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
VALIDATION_METRICS_FIELDS = {
    "schema_version",
    "selection_metric",
    "validation_metrics",
    "selected_model",
    "training_only_low_solar_threshold_kwh_m2_day",
}
DECISION_METRIC_INTEGER_COLUMNS = {
    "sample_count",
    "simulated_shortfall_day_count",
    "capacity_limited_shortfall_day_count",
}
DECISION_METRIC_NONNEGATIVE_COLUMNS = DECISION_METRIC_COLUMNS - {
    "scenario_id",
    "approach",
    "selected_model",
    "paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day",
    "paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day",
    "paired_mean_excess_difference_vs_persistence_kwh_day",
    "paired_mean_excess_difference_vs_seasonal_mean_kwh_day",
}


class ArtifactError(RuntimeError):
    """A user-actionable reason cached demonstration artifacts cannot be shown."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class DemoArtifacts:
    """Validated read-only inputs used by the demonstration."""

    forecasts: pd.DataFrame
    decisions: pd.DataFrame
    decision_metrics: pd.DataFrame
    manifest: dict[str, Any]
    quality: dict[str, Any]
    dates: tuple[date, ...]


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _development_data_hash(daily: pd.DataFrame) -> str:
    payload = daily.rename_axis("date").to_csv(index_label="date", lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _current_code_hashes(root: Path) -> dict[str, str]:
    sources = {
        "config": root / "src" / "sunreserve" / "config.py",
        "data": root / "src" / "sunreserve" / "data.py",
        "features": root / "src" / "sunreserve" / "features.py",
        "train": root / "src" / "sunreserve" / "train.py",
        "evaluate": root / "src" / "sunreserve" / "evaluate.py",
        "decision": root / "src" / "sunreserve" / "decision.py",
        "cli": root / "src" / "sunreserve" / "__main__.py",
        "inference": root / "src" / "sunreserve" / "inference.py",
        "source_provenance": root / "src" / "sunreserve" / "source_provenance.py",
    }
    result = {}
    for name, path in sources.items():
        if not path.is_file():
            raise ArtifactError("stale", f"Cannot verify validation code: {path} is missing.")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _experiment_directory(root: Path, experiment_id: str) -> Path:
    """Resolve one safe experiment directory without following directory symlinks."""
    if not isinstance(experiment_id, str) or _SAFE_EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ArtifactError("incompatible", "Experiment ID is not a safe run identifier.")
    experiments = root.resolve() / "artifacts" / "experiments"
    candidate = experiments / experiment_id
    if experiments.is_symlink() or candidate.is_symlink() or not candidate.is_dir():
        raise ArtifactError("missing", f"Validation experiment is missing: {experiment_id}.")
    try:
        resolved_experiments = experiments.resolve()
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != resolved_experiments:
            raise ArtifactError("incompatible", "Experiment directory resolves outside its root.")
    except OSError as error:
        raise ArtifactError(
            "corrupt", f"Experiment directory cannot be resolved: {error}"
        ) from error
    return resolved_candidate


def _verified_experiment_directory(root: Path, experiment_id: str) -> Path:
    """Check the experiment envelope before exposing its immutable validation output."""
    directory = _experiment_directory(root, experiment_id)
    manifest_path = directory / "experiment_manifest.json"
    selection_path = directory / "validation" / "selection_manifest.json"
    model_path = directory / "models" / "ridge_model.json"
    required_paths = (
        directory / "config.json",
        selection_path,
        directory / "validation" / "validation_predictions.csv",
        directory / "validation" / "validation_metrics.json",
        directory / "validation" / "decision_predictions.csv",
        directory / "validation" / "decision_metrics.csv",
        model_path,
    )
    if any(
        path.is_symlink() or not path.is_file() for path in (manifest_path, *required_paths)
    ) or any(path.is_symlink() for path in (directory / "validation", directory / "models")):
        raise ArtifactError("missing", f"Experiment {experiment_id} is incomplete.")
    try:
        envelope = _parse_json_object(manifest_path.read_bytes(), "Experiment manifest")
        selection_bytes = selection_path.read_bytes()
        model_bytes = model_path.read_bytes()
        selection = _parse_json_object(selection_bytes, "Experiment selection manifest")
    except OSError as error:
        raise ArtifactError(
            "corrupt", f"Experiment {experiment_id} cannot be read: {error}"
        ) from error
    if (
        type(envelope.get("schema_version")) is not int
        or envelope.get("schema_version") != 1
        or envelope.get("artifact_type") != "sunreserve_validation_experiment"
        or envelope.get("run_id") != experiment_id
        or envelope.get("model_path")
        != f"artifacts/experiments/{experiment_id}/models/ridge_model.json"
        or type(selection.get("schema_version")) is not int
        or selection.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or selection.get("artifact_type") != "sunreserve_validation_selection"
        or not isinstance(selection.get("status"), str)
        or not selection["status"].startswith("validation_only")
        or not isinstance(envelope.get("model_sha256"), str)
        or not isinstance(envelope.get("selection_manifest_sha256"), str)
        or not isinstance(selection.get("model_bundle"), dict)
        or selection["model_bundle"].get("path")
        != f"artifacts/experiments/{experiment_id}/models/ridge_model.json"
        or selection["model_bundle"].get("sha256") != envelope.get("model_sha256")
    ):
        raise ArtifactError("incompatible", f"Experiment {experiment_id} has an invalid manifest.")
    if (
        hashlib.sha256(model_bytes).hexdigest() != envelope["model_sha256"].lower()
        or hashlib.sha256(selection_bytes).hexdigest()
        != envelope["selection_manifest_sha256"].lower()
    ):
        raise ArtifactError(
            "stale", f"Experiment {experiment_id} checksum does not match its manifest."
        )
    return directory


def list_validation_experiments(root: Path) -> list[str]:
    """List only safe, complete experiments whose envelope checksums verify."""
    experiments = root.resolve() / "artifacts" / "experiments"
    if not experiments.is_dir() or experiments.is_symlink():
        return []
    try:
        candidates = sorted(experiments.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    available = []
    for candidate in candidates:
        if candidate.name.startswith(".") or _SAFE_EXPERIMENT_ID.fullmatch(candidate.name) is None:
            continue
        try:
            _verified_experiment_directory(root, candidate.name)
        except (ArtifactError, OSError):
            continue
        available.append(candidate.name)
    return available


def artifact_signature(
    root: Path,
    *,
    code_root: Path | None = None,
    experiment_id: str | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return stable content signatures to refresh Streamlit's cache when inputs change."""
    paths = ProjectPaths(root)
    validation_dir = (
        _verified_experiment_directory(root, experiment_id) / "validation"
        if experiment_id is not None
        else paths.validation_dir
    )
    candidates = [
        validation_dir / name
        for name in (
            "selection_manifest.json",
            "validation_predictions.csv",
            "validation_metrics.json",
            "decision_predictions.csv",
            "decision_metrics.csv",
        )
    ] + [
        paths.quality_path,
        paths.sidecar_path,
        paths.raw_path,
        paths.daily_path,
        paths.model_path,
    ]
    if experiment_id is not None:
        experiment_dir = _verified_experiment_directory(root, experiment_id)
        candidates.extend(
            [
                experiment_dir / "experiment_manifest.json",
                experiment_dir / "models" / "ridge_model.json",
            ]
        )
    source_root = (code_root or root).resolve()
    candidates.extend(
        [
            source_root / "src" / "sunreserve" / file
            for file in (
                "config.py",
                "data.py",
                "features.py",
                "train.py",
                "evaluate.py",
                "decision.py",
                "__main__.py",
                "inference.py",
                "source_provenance.py",
                "presentation.py",
            )
        ]
    )
    signature = []
    for path in candidates:
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            signature.append((str(path), f"sha256:{digest.hexdigest()}"))
        except (FileNotFoundError, NotADirectoryError):
            signature.append((str(path), "missing"))
        except OSError:
            signature.append((str(path), "unreadable"))
    return tuple(signature)


def _parse_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numbers must be finite")
        return number

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ArtifactError("corrupt", f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError("corrupt", f"{label} must contain a JSON object.")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ArtifactError("corrupt", f"{label} cannot be read: {error}") from error
    return _parse_json_object(payload, label)


def _read_csv(payload: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload))
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        raise ArtifactError("corrupt", f"{label} cannot be read: {error}") from error


def _require_columns(
    frame: pd.DataFrame, required: set[str], label: str, *, exact: bool = True
) -> None:
    if frame.columns.has_duplicates:
        raise ArtifactError("incompatible", f"{label} contains duplicate columns.")
    missing = required - set(frame.columns)
    if missing:
        raise ArtifactError(
            "incompatible",
            f"{label} is missing required fields: {', '.join(sorted(missing))}.",
        )
    if exact:
        unexpected = set(frame.columns) - required
        if unexpected:
            raise ArtifactError(
                "incompatible",
                f"{label} contains unsupported fields: {', '.join(sorted(unexpected))}.",
            )


def _require_exact_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unsupported {', '.join(sorted(extra))}")
        raise ArtifactError("incompatible", f"{label} has an invalid schema: {'; '.join(details)}.")


def _require_fields(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    if missing:
        raise ArtifactError(
            "incompatible", f"{label} is missing required fields: {', '.join(sorted(missing))}."
        )


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise ArtifactError("corrupt", f"{label} field {column} is not numeric.") from error
        if not np.isfinite(values).all():
            raise ArtifactError("corrupt", f"{label} field {column} contains non-finite values.")


def _finite_json_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError("incompatible", f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactError("corrupt", f"{label} must be finite.")
    if nonnegative and number < 0:
        raise ArtifactError("incompatible", f"{label} must be nonnegative.")
    return number


def _compare_json_values(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ArtifactError("stale", f"{label} fields do not match their source output.")
        for key in actual:
            _compare_json_values(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ArtifactError("stale", f"{label} rows do not match their source output.")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _compare_json_values(actual_item, expected_item, f"{label}[{index}]")
        return
    if actual is None or expected is None:
        if actual is not expected:
            raise ArtifactError("stale", f"{label} does not match its source output.")
        return
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise ArtifactError("stale", f"{label} does not match its source output.")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10):
            raise ArtifactError("stale", f"{label} does not match its source output.")
        return
    if actual != expected:
        raise ArtifactError("stale", f"{label} does not match its source output.")


def _validate_forecast_metrics(
    metrics: Any,
    forecasts: pd.DataFrame,
    *,
    expected_dates: pd.DatetimeIndex,
) -> None:
    if not isinstance(metrics, dict) or set(metrics) != set(PREDICTION_COLUMNS):
        raise ArtifactError("incompatible", "Forecast metrics must include all three models.")

    month_sample_counts = {
        f"{month:02d}": int((expected_dates.month == month).sum()) for month in range(1, 13)
    }
    season_months = {
        "DJF": {12, 1, 2},
        "MAM": {3, 4, 5},
        "JJA": {6, 7, 8},
        "SON": {9, 10, 11},
    }
    season_sample_counts = {
        season: int(expected_dates.month.isin(months).sum())
        for season, months in season_months.items()
    }

    for approach, prediction_column in PREDICTION_COLUMNS.items():
        item = metrics[approach]
        if not isinstance(item, dict):
            raise ArtifactError("incompatible", f"{approach} metrics must be a JSON object.")
        _require_exact_keys(item, FORECAST_METRIC_FIELDS, f"{approach} metrics")
        if type(item["sample_count"]) is not int or item["sample_count"] != len(expected_dates):
            raise ArtifactError("incompatible", f"{approach} metrics have an invalid sample count.")
        for field in ("mae", "rmse"):
            _finite_json_number(item[field], f"{approach} {field}", nonnegative=True)
        _finite_json_number(item["bias_prediction_minus_actual"], f"{approach} bias")
        if item["prediction_clip_rule"] != PREDICTION_CLIP_RULE:
            raise ArtifactError(
                "incompatible", f"{approach} prediction clipping rule is unsupported."
            )

        for field, expected_counts in (
            ("monthly_mae_by_target_calendar_month", month_sample_counts),
            ("seasonal_mae", season_sample_counts),
        ):
            values = item[field]
            if not isinstance(values, dict) or set(values) != set(expected_counts):
                raise ArtifactError("incompatible", f"{approach} {field} has an invalid schema.")
            for key, number in values.items():
                _finite_json_number(number, f"{approach} {field}.{key}", nonnegative=True)

        for field, expected_counts in (
            ("monthly_sample_count", month_sample_counts),
            ("seasonal_sample_count", season_sample_counts),
        ):
            counts = item[field]
            if not isinstance(counts, dict) or counts != expected_counts:
                raise ArtifactError("incompatible", f"{approach} {field} does not match 2022.")
            if any(type(count) is not int or count <= 0 for count in counts.values()):
                raise ArtifactError("incompatible", f"{approach} {field} contains invalid counts.")

        low_solar = item["low_solar"]
        if not isinstance(low_solar, dict):
            raise ArtifactError("incompatible", f"{approach} low_solar metrics must be an object.")
        _require_exact_keys(
            low_solar,
            {
                "threshold_kwh_m2_day",
                "sample_count",
                "mae",
                "rmse",
                "bias_prediction_minus_actual",
            },
            f"{approach} low_solar metrics",
        )
        threshold = _finite_json_number(
            low_solar["threshold_kwh_m2_day"], f"{approach} low-solar threshold", nonnegative=True
        )
        low_count = low_solar["sample_count"]
        if type(low_count) is not int or not 0 <= low_count <= len(expected_dates):
            raise ArtifactError("incompatible", f"{approach} low_solar sample count is invalid.")
        if low_count == 0:
            if any(
                low_solar[field] is not None
                for field in ("mae", "rmse", "bias_prediction_minus_actual")
            ):
                raise ArtifactError(
                    "incompatible", f"{approach} empty low_solar metrics must be null."
                )
        else:
            for field in ("mae", "rmse"):
                _finite_json_number(
                    low_solar[field], f"{approach} low_solar {field}", nonnegative=True
                )
            _finite_json_number(
                low_solar["bias_prediction_minus_actual"], f"{approach} low_solar bias"
            )

        expected = forecast_metrics(
            forecasts[RADIATION_COLUMN],
            forecasts[prediction_column],
            forecasts["target_date"],
            low_solar_threshold=threshold,
        )
        _compare_json_values(item, expected, f"{approach} validation metrics")


def _decision_scenario_settings(
    manifest: dict[str, Any], config: ExperimentConfig
) -> tuple[dict[str, tuple[float, float, float]], list[dict[str, Any]]]:
    simulation = manifest.get("decision_simulation")
    if not isinstance(simulation, dict):
        raise ArtifactError("incompatible", "Decision simulation metadata must be an object.")
    if simulation.get("partition") != "validation":
        raise ArtifactError("incompatible", "Decision outputs must use the validation partition.")
    if simulation.get("approaches") != list(APPROACHES):
        raise ArtifactError("incompatible", "Decision simulation approaches are unsupported.")

    primary = simulation.get("primary_scenario")
    primary_settings = {
        "panel_area_m2": config.panel_area_m2,
        "efficiency": config.conversion_efficiency,
        "derating": config.derating_factor,
        "demand_kwh": config.daily_demand_kwh,
        "capacity_kwh": config.reserve_cap_kwh,
        "buffer_kwh": config.reserve_buffer_kwh,
    }
    if not isinstance(primary, dict):
        raise ArtifactError("incompatible", "Primary reserve scenario must be an object.")
    _require_exact_keys(primary, set(primary_settings), "Primary reserve scenario")
    for name, expected in primary_settings.items():
        actual = _finite_json_number(
            primary[name], f"Primary reserve scenario {name}", nonnegative=True
        )
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ArtifactError("stale", f"Primary reserve scenario {name} does not match config.")

    grid = simulation.get("sensitivity_grid")
    grid_fields = {
        "demand_kwh",
        "capacity_kwh",
        "buffer_fraction_of_demand",
        "scenario_count",
    }
    if not isinstance(grid, dict):
        raise ArtifactError("incompatible", "Reserve sensitivity grid must be an object.")
    _require_exact_keys(grid, grid_fields, "Reserve sensitivity grid")
    expected_grid = {
        "demand_kwh": list(DEMAND_LEVELS_KWH),
        "capacity_kwh": list(CAPACITY_LEVELS_KWH),
        "buffer_fraction_of_demand": list(BUFFER_FRACTIONS),
    }
    for name, expected_values in expected_grid.items():
        values = grid[name]
        if not isinstance(values, list) or len(values) != len(expected_values):
            raise ArtifactError("incompatible", f"Reserve sensitivity grid {name} is invalid.")
        normalized = [
            _finite_json_number(value, f"Reserve sensitivity grid {name}", nonnegative=True)
            for value in values
        ]
        if not np.allclose(normalized, expected_values, rtol=0, atol=1e-12):
            raise ArtifactError("incompatible", f"Reserve sensitivity grid {name} is unsupported.")
    expected_scenario_count = (
        len(DEMAND_LEVELS_KWH) * len(CAPACITY_LEVELS_KWH) * len(BUFFER_FRACTIONS)
    )
    if type(grid["scenario_count"]) is not int or grid["scenario_count"] != expected_scenario_count:
        raise ArtifactError("incompatible", "Reserve sensitivity scenario count is invalid.")

    settings = {
        "primary": (
            float(config.daily_demand_kwh),
            float(config.reserve_cap_kwh),
            float(config.reserve_buffer_kwh),
        )
    }
    for demand in DEMAND_LEVELS_KWH:
        for capacity in CAPACITY_LEVELS_KWH:
            for fraction in BUFFER_FRACTIONS:
                scenario_id = f"grid_d{demand:g}_c{capacity:g}_b{fraction:g}"
                settings[scenario_id] = (demand, capacity, demand * fraction)

    records = simulation.get("metrics")
    if not isinstance(records, list):
        raise ArtifactError("incompatible", "Decision simulation metrics must be a JSON array.")
    return settings, records


def _validate_decision_summaries(
    decision_metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    scenario_settings: dict[str, tuple[float, float, float]],
    manifest_records: list[dict[str, Any]],
    selected_model: str,
    expected_dates: pd.DatetimeIndex,
) -> None:
    expected_scenarios = set(scenario_settings)
    expected_approaches = set(APPROACHES)
    expected_metric_keys = {
        (scenario, approach) for scenario in expected_scenarios for approach in expected_approaches
    }
    metric_keys = set(
        zip(decision_metrics["scenario_id"], decision_metrics["approach"], strict=True)
    )
    if metric_keys != expected_metric_keys:
        raise ArtifactError(
            "incompatible", "Decision metrics do not cover every scenario/approach."
        )
    if len(decision_metrics) != len(expected_metric_keys):
        raise ArtifactError(
            "incompatible", "Decision metrics contain duplicate scenario/approach rows."
        )

    for scenario, expected_settings in scenario_settings.items():
        block = decisions.loc[decisions["scenario_id"] == scenario]
        if block.empty:
            raise ArtifactError("incompatible", f"Decision predictions omit scenario {scenario}.")
        for column, expected_value in zip(
            ("demand_kwh", "capacity_kwh", "buffer_kwh"), expected_settings, strict=True
        ):
            values = pd.to_numeric(block[column], errors="raise").to_numpy(dtype=float)
            if not np.allclose(values, expected_value, rtol=0, atol=1e-10):
                raise ArtifactError(
                    "stale", f"Decision scenario {scenario} has inconsistent {column}."
                )
        if len(block) != len(expected_dates) * len(APPROACHES):
            raise ArtifactError(
                "incompatible", f"Decision scenario {scenario} has an invalid sample matrix."
            )

    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in manifest_records:
        if not isinstance(record, dict):
            raise ArtifactError(
                "incompatible", "Decision simulation metric entries must be objects."
            )
        _require_exact_keys(record, DECISION_METRIC_COLUMNS, "Decision simulation metric entry")
        key = (record["scenario_id"], record["approach"])
        if key in records_by_key:
            raise ArtifactError(
                "incompatible", "Decision simulation metrics contain duplicate rows."
            )
        records_by_key[key] = record
    if set(records_by_key) != expected_metric_keys:
        raise ArtifactError(
            "incompatible", "Manifest decision metrics omit scenario/approach rows."
        )

    for row in decision_metrics.itertuples(index=False):
        key = (row.scenario_id, row.approach)
        if row.selected_model != selected_model:
            raise ArtifactError("stale", "Decision metrics and selected model disagree.")
        if (
            isinstance(row.sample_count, (bool, np.bool_))
            or not isinstance(row.sample_count, (int, np.integer))
            or row.sample_count != len(expected_dates)
        ):
            raise ArtifactError(
                "incompatible", f"Decision metrics {key} has an invalid sample count."
            )
        if any(
            isinstance(getattr(row, column), (bool, np.bool_))
            or not isinstance(getattr(row, column), (int, np.integer))
            or getattr(row, column) < 0
            for column in DECISION_METRIC_INTEGER_COLUMNS - {"sample_count"}
        ):
            raise ArtifactError("incompatible", f"Decision metrics {key} has invalid event counts.")
        share = float(row.simulated_shortfall_day_share)
        if not 0 <= share <= 1:
            raise ArtifactError("incompatible", f"Decision metrics {key} has an invalid day share.")

        scenario_rows = decisions.loc[decisions["scenario_id"] == row.scenario_id]
        sample = scenario_rows.loc[scenario_rows["approach"] == row.approach]
        if len(sample) != len(expected_dates):
            raise ArtifactError(
                "incompatible", f"Decision output {key} has an invalid sample count."
            )
        shortfall = sample["simulated_shortfall_kwh"].to_numpy(dtype=float)
        excess = sample["excess_prepared_reserve_kwh"].to_numpy(dtype=float)
        capacity_floor = sample["capacity_limited_shortfall_kwh"].to_numpy(dtype=float)
        additional = sample["additional_shortfall_above_capacity_floor_kwh"].to_numpy(dtype=float)
        means = {
            approach: {
                "shortfall": float(
                    scenario_rows.loc[
                        scenario_rows["approach"] == approach, "simulated_shortfall_kwh"
                    ].mean()
                ),
                "excess": float(
                    scenario_rows.loc[
                        scenario_rows["approach"] == approach, "excess_prepared_reserve_kwh"
                    ].mean()
                ),
            }
            for approach in APPROACHES
        }
        expected_values = {
            "sample_count": len(sample),
            "simulated_shortfall_day_count": int(np.count_nonzero(shortfall > 0)),
            "simulated_shortfall_day_share": float(np.mean(shortfall > 0)),
            "total_simulated_shortfall_kwh": float(shortfall.sum()),
            "mean_simulated_shortfall_kwh_day": float(shortfall.mean()),
            "total_excess_prepared_reserve_kwh": float(excess.sum()),
            "mean_excess_prepared_reserve_kwh_day": float(excess.mean()),
            "capacity_limited_shortfall_day_count": int(np.count_nonzero(capacity_floor > 0)),
            "total_capacity_limited_shortfall_kwh": float(capacity_floor.sum()),
            "mean_capacity_limited_shortfall_kwh_day": float(capacity_floor.mean()),
            "total_additional_shortfall_above_capacity_floor_kwh": float(additional.sum()),
            "mean_additional_shortfall_above_capacity_floor_kwh_day": float(additional.mean()),
            "paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day": (
                means[row.approach]["shortfall"] - means["persistence"]["shortfall"]
            ),
            "paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day": (
                means[row.approach]["shortfall"] - means["seasonal_mean"]["shortfall"]
            ),
            "paired_mean_excess_difference_vs_persistence_kwh_day": (
                means[row.approach]["excess"] - means["persistence"]["excess"]
            ),
            "paired_mean_excess_difference_vs_seasonal_mean_kwh_day": (
                means[row.approach]["excess"] - means["seasonal_mean"]["excess"]
            ),
        }
        for column, expected in expected_values.items():
            if not math.isclose(float(getattr(row, column)), expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ArtifactError(
                    "stale", f"Decision metric {key}.{column} does not match predictions."
                )
            manifest_value = records_by_key[key][column]
            if isinstance(expected, int):
                if type(manifest_value) is not int or manifest_value != expected:
                    raise ArtifactError(
                        "stale", f"Manifest decision metric {key}.{column} disagrees."
                    )
            elif not math.isclose(
                _finite_json_number(manifest_value, f"Manifest decision metric {key}.{column}"),
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ArtifactError("stale", f"Manifest decision metric {key}.{column} disagrees.")
        if records_by_key[key]["selected_model"] != row.selected_model:
            raise ArtifactError("stale", f"Manifest decision metric {key} has a different model.")


def _load_development_daily(paths: ProjectPaths, config: ExperimentConfig) -> pd.DataFrame:
    for path, label in (
        (paths.quality_path, "data quality report"),
        (paths.sidecar_path, "source provenance sidecar"),
        (paths.raw_path, "raw source cache"),
        (paths.daily_path, "prepared daily data"),
    ):
        if not path.is_file():
            raise ArtifactError(
                "missing",
                f"{label} is missing at `{path}`. Restore local cached inputs, then run "
                "`python -m sunreserve prepare` and `python -m sunreserve validate`.",
            )

    quality = _read_json(paths.quality_path, "data quality report")
    sidecar = _read_json(paths.sidecar_path, "source provenance sidecar")
    try:
        raw_digest = hashlib.sha256(paths.raw_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactError("corrupt", f"Raw source cache cannot be read: {error}") from error
    if raw_digest != sidecar.get("sha256") or raw_digest != quality.get("raw_sha256"):
        raise ArtifactError(
            "stale", "Raw source checksum does not match the prepared quality report."
        )
    if sidecar.get("request_url") != request_url(config):
        raise ArtifactError(
            "stale", "Source provenance does not match the saved experiment config."
        )
    if (
        quality.get("requested_start") != config.start_date.isoformat()
        or quality.get("requested_end") != config.end_date.isoformat()
    ):
        raise ArtifactError("stale", "Prepared data date range does not match the saved config.")

    development_count = (config.validation_end - config.start_date).days + 1
    expected = pd.date_range(config.start_date, config.validation_end, freq="D")
    try:
        daily = pd.read_csv(paths.daily_path, nrows=development_count, parse_dates=["date"])
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise ArtifactError(
            "corrupt", f"Prepared development rows cannot be read: {error}"
        ) from error
    required = {"date", *config.parameters}
    missing = required - set(daily.columns)
    if missing or len(daily) != len(expected):
        raise ArtifactError(
            "incompatible", "Prepared development data has an incompatible schema or length."
        )
    dates = pd.DatetimeIndex(daily.pop("date"))
    if not dates.equals(expected):
        raise ArtifactError(
            "stale", "Prepared development rows are not the expected daily calendar."
        )
    daily.index = dates
    daily.index.name = "date"
    return daily


def load_demo_artifacts(
    root: Path,
    *,
    code_root: Path | None = None,
    experiment_id: str | None = None,
) -> DemoArtifacts:
    """Load and verify cached validation artifacts without fetching or fitting models."""
    root = root.resolve()
    paths = ProjectPaths(root)
    validation_dir = (
        _verified_experiment_directory(root, experiment_id) / "validation"
        if experiment_id is not None
        else paths.validation_dir
    )
    required_files = {
        "selection_manifest.json": validation_dir / "selection_manifest.json",
        "validation_predictions.csv": validation_dir / "validation_predictions.csv",
        "decision_predictions.csv": validation_dir / "decision_predictions.csv",
        "decision_metrics.csv": validation_dir / "decision_metrics.csv",
        "validation_metrics.json": validation_dir / "validation_metrics.json",
    }
    output_labels = {
        "selection_manifest.json": "selection manifest",
        "validation_predictions.csv": "validation predictions",
        "decision_predictions.csv": "decision predictions",
        "decision_metrics.csv": "decision metrics",
        "validation_metrics.json": "validation metrics",
    }
    missing = [
        f"{output_labels[name]} (`{path}`)"
        for name, path in required_files.items()
        if not path.is_file()
    ]
    if missing:
        raise ArtifactError(
            "missing",
            "Missing cached validation output: "
            + "; ".join(missing)
            + ". Restore the shipped artifacts or create and select a validation experiment.",
        )

    try:
        output_bytes = {name: path.read_bytes() for name, path in required_files.items()}
    except FileNotFoundError as error:
        raise ArtifactError(
            "missing",
            "A cached validation output disappeared while loading. Restore the shipped artifacts "
            "or create and select a validation experiment.",
        ) from error
    except OSError as error:
        raise ArtifactError(
            "corrupt", f"A cached validation output cannot be read: {error}"
        ) from error

    manifest = _parse_json_object(output_bytes["selection_manifest.json"], "Selection manifest")
    metrics_payload = _parse_json_object(
        output_bytes["validation_metrics.json"], "Validation metrics"
    )
    _require_fields(manifest, MANIFEST_REQUIRED_FIELDS, "Selection manifest")
    _require_exact_keys(metrics_payload, VALIDATION_METRICS_FIELDS, "Validation metrics")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("artifact_type") != "sunreserve_validation_selection"
    ):
        raise ArtifactError(
            "incompatible", "Selection manifest schema or artifact type is unsupported."
        )
    if (
        type(metrics_payload.get("schema_version")) is not int
        or metrics_payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION
    ):
        raise ArtifactError("incompatible", "Validation metrics schema version is unsupported.")
    if not isinstance(manifest.get("status"), str) or not manifest["status"].startswith(
        "validation_only"
    ):
        raise ArtifactError(
            "incompatible", "Only validation-only artifacts can be shown in this demo."
        )
    expected_output_hashes = manifest.get("development_artifact_hashes")
    if not isinstance(expected_output_hashes, dict) or set(expected_output_hashes) != (
        CHECKSUM_BOUND_OUTPUTS
    ):
        raise ArtifactError("incompatible", "Manifest lacks required development artifact hashes.")
    for name, expected_hash in expected_output_hashes.items():
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash) is None
        ):
            raise ArtifactError("incompatible", f"Manifest checksum for {name} is invalid.")
        actual_hash = hashlib.sha256(output_bytes[name]).hexdigest()
        if actual_hash != expected_hash.lower():
            raise ArtifactError("stale", f"Validation artifact checksum does not match: {name}.")

    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ArtifactError("incompatible", "Saved experiment configuration must be an object.")
    _require_exact_keys(configuration, set(ExperimentConfig().to_dict()), "Saved configuration")
    try:
        config = ExperimentConfig.from_dict(configuration)
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(
            "incompatible", f"Saved experiment configuration is invalid: {error}"
        ) from error
    configuration_hash = manifest.get("configuration_sha256")
    if (
        not isinstance(configuration_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", configuration_hash) is None
        or configuration_hash.lower() != _canonical_hash(config.to_dict())
    ):
        raise ArtifactError("stale", "Saved configuration checksum does not match the manifest.")
    if manifest.get("target") != TARGET_COLUMN or manifest.get("target_units") != "kWh/m^2/day":
        raise ArtifactError("incompatible", "Validation target or units are unsupported.")
    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ArtifactError("incompatible", "Selection manifest input hashes must be an object.")
    _require_exact_keys(input_hashes, INPUT_HASH_FIELDS, "Selection input hashes")
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
        for value in input_hashes.values()
    ):
        raise ArtifactError("incompatible", "Selection manifest contains an invalid input hash.")
    if manifest.get("validation_target_period") != [
        config.validation_start.isoformat(),
        config.validation_end.isoformat(),
    ]:
        raise ArtifactError(
            "incompatible", "Saved validation period does not match its configuration."
        )
    if config.validation_start != date(2022, 1, 1) or config.validation_end != date(2022, 12, 31):
        raise ArtifactError(
            "incompatible", "This demonstration is limited to the 2022 validation period."
        )

    model_selection = manifest.get("model_selection")
    if not isinstance(model_selection, dict):
        raise ArtifactError("incompatible", "Model-selection metadata must be an object.")
    _require_fields(
        model_selection,
        {
            "metric",
            "selection_partition",
            "tie_order",
            "selected_model",
            "validation_metrics",
            "prediction_clip_rule",
            "training_only_low_solar_threshold_kwh_m2_day",
            "low_solar_quantile",
            "quantile_method",
            "validation_refit",
        },
        "Model-selection metadata",
    )
    if (
        model_selection["metric"] != config.primary_metric
        or model_selection["selection_partition"] != "validation"
        or model_selection["tie_order"] != list(PREDICTION_COLUMNS)
        or model_selection["prediction_clip_rule"] != PREDICTION_CLIP_RULE
        or model_selection["quantile_method"] != "numpy linear"
        or model_selection["validation_refit"] is not False
    ):
        raise ArtifactError("incompatible", "Model-selection schema or policy is unsupported.")
    low_solar_threshold = _finite_json_number(
        model_selection["training_only_low_solar_threshold_kwh_m2_day"],
        "Training-only low-solar threshold",
        nonnegative=True,
    )
    low_solar_quantile = _finite_json_number(
        model_selection["low_solar_quantile"], "Low-solar quantile", nonnegative=True
    )
    if not 0 < low_solar_quantile < 1 or not math.isclose(
        low_solar_quantile, config.low_solar_quantile, rel_tol=0, abs_tol=1e-12
    ):
        raise ArtifactError("incompatible", "Model-selection low-solar quantile is unsupported.")

    producer_code_hashes = manifest.get("code_hashes")
    if (
        not isinstance(producer_code_hashes, dict)
        or not producer_code_hashes
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            for name, digest in producer_code_hashes.items()
        )
        or not isinstance(manifest.get("code_revision"), str)
    ):
        raise ArtifactError("incompatible", "Selection manifest producer identity is invalid.")
    daily = _load_development_daily(paths, config)
    expected_data_hash = _development_data_hash(daily)
    quality = _read_json(paths.quality_path, "data quality report")
    sidecar = _read_json(paths.sidecar_path, "source provenance sidecar")
    if input_hashes.get("prepared_development_rows_sha256") != expected_data_hash:
        raise ArtifactError("stale", "Prepared validation inputs have changed since the run.")
    if input_hashes.get("raw_response_sha256") != sidecar.get("sha256"):
        raise ArtifactError(
            "stale", "Validation artifacts reference a different raw source response."
        )
    if input_hashes["prepared_daily_csv_sha256"] != quality.get("prepared_daily_csv_sha256"):
        raise ArtifactError(
            "stale", "Validation artifacts reference different prepared daily data."
        )
    if (
        input_hashes.get("quality_report_sha256")
        != hashlib.sha256(paths.quality_path.read_bytes()).hexdigest()
    ):
        raise ArtifactError("stale", "Quality report changed after validation was frozen.")

    forecasts = _read_csv(output_bytes["validation_predictions.csv"], "Validation predictions")
    decisions = _read_csv(output_bytes["decision_predictions.csv"], "Decision predictions")
    decision_metrics = _read_csv(output_bytes["decision_metrics.csv"], "Decision metrics")
    _require_columns(forecasts, FORECAST_COLUMNS, "Validation predictions")
    _require_columns(decisions, DECISION_COLUMNS, "Decision predictions")
    _require_columns(decision_metrics, DECISION_METRIC_COLUMNS, "Decision metrics")
    if forecasts.empty or decisions.empty or decision_metrics.empty:
        raise ArtifactError("incompatible", "Validation outputs contain no rows.")
    _finite_columns(
        decision_metrics,
        tuple(DECISION_METRIC_COLUMNS - {"scenario_id", "approach", "selected_model"}),
        "Decision metrics",
    )
    if (decision_metrics[list(DECISION_METRIC_NONNEGATIVE_COLUMNS)] < 0).any().any():
        raise ArtifactError("incompatible", "Decision metrics contain negative quantities.")

    for frame, label in (
        (forecasts, "Validation predictions"),
        (decisions, "Decision predictions"),
    ):
        try:
            frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
        except (OverflowError, TypeError, ValueError) as error:
            raise ArtifactError("corrupt", f"{label} contains an invalid target date.") from error
        if frame["target_date"].isna().any():
            raise ArtifactError("corrupt", f"{label} contains a missing target date.")
    try:
        forecasts["issue_date"] = pd.to_datetime(forecasts["issue_date"], errors="raise")
    except (OverflowError, TypeError, ValueError) as error:
        raise ArtifactError(
            "corrupt", "Validation predictions contain an invalid issue date."
        ) from error
    if (
        forecasts["target_date"].duplicated().any()
        or not forecasts["target_date"].is_monotonic_increasing
    ):
        raise ArtifactError("incompatible", "Validation predictions must have sorted unique dates.")
    expected_dates = pd.date_range(config.validation_start, config.validation_end, freq="D")
    if not pd.DatetimeIndex(forecasts["target_date"]).equals(expected_dates):
        raise ArtifactError("incompatible", "Validation prediction dates do not cover all of 2022.")
    if not np.array_equal(
        forecasts["issue_date"].to_numpy(),
        (forecasts["target_date"] - pd.Timedelta(days=1)).to_numpy(),
    ):
        raise ArtifactError(
            "incompatible", "Validation predictions must issue one day before each target."
        )
    _finite_columns(
        forecasts,
        (
            RADIATION_COLUMN,
            *PREDICTION_COLUMNS.values(),
        ),
        "Validation predictions",
    )
    if (forecasts[[RADIATION_COLUMN, *PREDICTION_COLUMNS.values()]] < 0).any().any():
        raise ArtifactError("incompatible", "Radiation values and predictions must be nonnegative.")

    selected_model = model_selection.get("selected_model")
    if not isinstance(selected_model, str) or selected_model not in PREDICTION_COLUMNS:
        raise ArtifactError("incompatible", "Manifest names an unsupported selected model.")
    if (
        metrics_payload.get("selection_metric") != config.primary_metric
        or metrics_payload.get("selected_model") != selected_model
    ):
        raise ArtifactError("stale", "Validation metrics and selection manifest disagree.")
    metrics_threshold = _finite_json_number(
        metrics_payload["training_only_low_solar_threshold_kwh_m2_day"],
        "Validation-metrics low-solar threshold",
        nonnegative=True,
    )
    if not math.isclose(metrics_threshold, low_solar_threshold, rel_tol=0, abs_tol=1e-12):
        raise ArtifactError("stale", "Validation metrics use a different low-solar threshold.")
    metrics_by_model = metrics_payload["validation_metrics"]
    _compare_json_values(
        model_selection["validation_metrics"],
        metrics_by_model,
        "Manifest and validation-file forecast metrics",
    )
    _validate_forecast_metrics(
        metrics_by_model,
        forecasts,
        expected_dates=expected_dates,
    )
    if not math.isclose(
        _finite_json_number(
            metrics_by_model[selected_model]["low_solar"]["threshold_kwh_m2_day"],
            "Selected model low-solar threshold",
            nonnegative=True,
        ),
        low_solar_threshold,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ArtifactError("stale", "Selected-model metrics use a different low-solar threshold.")

    try:
        decisions["scenario_id"] = decisions["scenario_id"].astype(str)
        decisions["approach"] = decisions["approach"].astype(str)
        decision_metrics["scenario_id"] = decision_metrics["scenario_id"].astype(str)
        decision_metrics["approach"] = decision_metrics["approach"].astype(str)
        scenarios = set(decisions["scenario_id"])
        approaches = set(decisions["approach"])
        decision_dates = pd.DatetimeIndex(decisions["target_date"])
    except (TypeError, ValueError) as error:
        raise ArtifactError("corrupt", "Decision output keys are invalid.") from error
    if approaches != set(APPROACHES):
        raise ArtifactError(
            "incompatible", "Decision outputs must contain all three models and backup reference."
        )
    scenario_settings, manifest_decision_records = _decision_scenario_settings(manifest, config)
    if scenarios != set(scenario_settings):
        raise ArtifactError("incompatible", "Decision scenario IDs do not match the declared grid.")
    expected_scenarios = len(scenario_settings)
    expected_count = len(expected_dates) * expected_scenarios * len(APPROACHES)
    if len(decisions) != expected_count:
        raise ArtifactError(
            "incompatible", "Decision outputs have an incomplete scenario/date matrix."
        )
    decision_key = ["scenario_id", "approach", "target_date"]
    if decisions.duplicated(decision_key).any():
        raise ArtifactError(
            "incompatible", "Decision outputs contain duplicate scenario/date rows."
        )
    if set(decision_dates) != set(expected_dates):
        raise ArtifactError(
            "incompatible", "Decision outputs do not match the validation target dates."
        )
    expected_decision_keys = {
        (scenario, approach, target_date)
        for scenario in scenario_settings
        for approach in APPROACHES
        for target_date in expected_dates
    }
    actual_decision_keys = set(
        zip(decisions["scenario_id"], decisions["approach"], decision_dates, strict=True)
    )
    if actual_decision_keys != expected_decision_keys:
        raise ArtifactError(
            "incompatible", "Decision outputs have incomplete date/approach coverage."
        )

    numeric_decision_columns = tuple(DECISION_COLUMNS - {"scenario_id", "approach", "target_date"})
    optional_backup_columns = {
        "predicted_radiation_kwh_m2_day",
        "predicted_generation_proxy_kwh_day",
    }
    for approach in APPROACHES:
        selected = decisions.loc[decisions["approach"] == approach]
        fields = tuple(
            name
            for name in numeric_decision_columns
            if approach != "always_prepare_backup" or name not in optional_backup_columns
        )
        _finite_columns(selected, fields, "Decision predictions")
        if (
            approach == "always_prepare_backup"
            and not selected[list(optional_backup_columns)].isna().all().all()
        ):
            raise ArtifactError(
                "incompatible", "Backup-reference predictions must be explicitly unavailable."
            )
    nonnegative_decision_columns = tuple(set(numeric_decision_columns) - optional_backup_columns)
    if (decisions[list(nonnegative_decision_columns)] < 0).any().any():
        raise ArtifactError("incompatible", "Decision quantities must be nonnegative.")
    _validate_decision_summaries(
        decision_metrics,
        decisions,
        scenario_settings=scenario_settings,
        manifest_records=manifest_decision_records,
        selected_model=selected_model,
        expected_dates=expected_dates,
    )
    invariants = (
        "actual_radiation_kwh_m2_day",
        "demand_kwh",
        "capacity_kwh",
        "buffer_kwh",
        "actual_generation_proxy_kwh_day",
        "capacity_limited_shortfall_kwh",
    )
    for (_, _), group in decisions.groupby(["scenario_id", "target_date"], sort=False):
        if any(group[column].nunique(dropna=False) != 1 for column in invariants):
            raise ArtifactError(
                "incompatible", "Approaches do not share identical observations/settings."
            )
    forecast_lookup = forecasts.set_index("target_date")
    for approach, column in PREDICTION_COLUMNS.items():
        subset = decisions.loc[decisions["approach"] == approach]
        expected_values = subset["target_date"].map(forecast_lookup[column]).to_numpy(dtype=float)
        actual_values = pd.to_numeric(subset["predicted_radiation_kwh_m2_day"]).to_numpy(
            dtype=float
        )
        if not np.allclose(expected_values, actual_values, rtol=0, atol=1e-10):
            raise ArtifactError("stale", f"Decision rows do not match {approach} forecast outputs.")
        expected_actuals = (
            subset["target_date"].map(forecast_lookup[RADIATION_COLUMN]).to_numpy(dtype=float)
        )
        observed_actuals = subset["actual_radiation_kwh_m2_day"].to_numpy(dtype=float)
        if not np.allclose(expected_actuals, observed_actuals, rtol=0, atol=1e-10):
            raise ArtifactError(
                "stale", "Decision rows do not match observed validation radiation."
            )

    if set(decision_metrics["scenario_id"].astype(str)) != scenarios:
        raise ArtifactError(
            "incompatible", "Decision metrics and predictions contain different scenarios."
        )
    if set(decision_metrics["approach"].astype(str)) != set(APPROACHES):
        raise ArtifactError("incompatible", "Decision metrics omit a required approach.")
    if len(decision_metrics) != expected_scenarios * len(APPROACHES):
        raise ArtifactError(
            "incompatible", "Decision metrics have an incomplete scenario/approach matrix."
        )
    if decision_metrics.duplicated(["scenario_id", "approach"]).any():
        raise ArtifactError(
            "incompatible", "Decision metrics contain duplicate scenario/approach rows."
        )
    if set(decision_metrics["selected_model"].astype(str)) != {selected_model}:
        raise ArtifactError("stale", "Decision metrics and selected model disagree.")
    return DemoArtifacts(
        forecasts=forecasts,
        decisions=decisions,
        decision_metrics=decision_metrics,
        manifest=manifest,
        quality=quality,
        dates=tuple(timestamp.date() for timestamp in expected_dates),
    )


def failure_examples(forecasts: pd.DataFrame, model: str) -> pd.DataFrame:
    """Choose deterministic largest overprediction and underprediction examples."""
    column = PREDICTION_COLUMNS[model]
    frame = forecasts[["target_date", RADIATION_COLUMN, column]].copy()
    frame["prediction_error_kwh_m2_day"] = frame[column] - frame[RADIATION_COLUMN]
    examples = []
    over = frame.loc[frame["prediction_error_kwh_m2_day"] > 0].nlargest(
        1, "prediction_error_kwh_m2_day"
    )
    under = frame.loc[frame["prediction_error_kwh_m2_day"] < 0].nsmallest(
        1, "prediction_error_kwh_m2_day"
    )
    if not over.empty:
        over = over.assign(example_type="Largest overprediction")
        examples.append(over)
    if not under.empty:
        under = under.assign(example_type="Largest underprediction")
        examples.append(under)
    if not examples:
        return pd.DataFrame(columns=[*frame.columns, "example_type"])
    result = pd.concat(examples, ignore_index=True)
    result["target_date"] = result["target_date"].dt.date
    return result[
        ["example_type", "target_date", RADIATION_COLUMN, column, "prediction_error_kwh_m2_day"]
    ]
