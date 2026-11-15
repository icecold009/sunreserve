"""Offline artifact loading and integrity checks for the Streamlit demonstration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import ProjectPaths, request_url
from sunreserve.decision import APPROACHES, PREDICTION_COLUMNS, RADIATION_COLUMN
from sunreserve.evaluate import ARTIFACT_SCHEMA_VERSION

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
    "total_capacity_limited_shortfall_kwh",
    "total_additional_shortfall_above_capacity_floor_kwh",
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
    }
    result = {}
    for name, path in sources.items():
        if not path.is_file():
            raise ArtifactError("stale", f"Cannot verify validation code: {path} is missing.")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def artifact_signature(
    root: Path, *, code_root: Path | None = None
) -> tuple[tuple[str, int | None, int | None], ...]:
    """Return file metadata used to refresh Streamlit's cache when inputs change."""
    paths = ProjectPaths(root)
    candidates = [
        paths.validation_dir / name
        for name in (
            "selection_manifest.json",
            "validation_predictions.csv",
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
            )
        ]
    )
    signature = []
    for path in candidates:
        try:
            stat = path.stat()
            signature.append((str(path), stat.st_size, stat.st_mtime_ns))
        except OSError:
            signature.append((str(path), None, None))
    return tuple(signature)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError("corrupt", f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError("corrupt", f"{label} must contain a JSON object.")
    return value


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        raise ArtifactError("corrupt", f"{label} cannot be read: {error}") from error


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ArtifactError(
            "incompatible",
            f"{label} is missing required fields: {', '.join(sorted(missing))}.",
        )


def _finite_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ArtifactError("corrupt", f"{label} field {column} is not numeric.") from error
        if not np.isfinite(values).all():
            raise ArtifactError("corrupt", f"{label} field {column} contains non-finite values.")


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


def load_demo_artifacts(root: Path, *, code_root: Path | None = None) -> DemoArtifacts:
    """Load and verify cached validation artifacts without fetching or fitting models."""
    root = root.resolve()
    paths = ProjectPaths(root)
    required_files = {
        "selection manifest": paths.validation_dir / "selection_manifest.json",
        "validation predictions": paths.validation_dir / "validation_predictions.csv",
        "decision predictions": paths.validation_dir / "decision_predictions.csv",
        "decision metrics": paths.validation_dir / "decision_metrics.csv",
        "validation metrics": paths.validation_dir / "validation_metrics.json",
    }
    missing = [
        f"{label} (`{path}`)" for label, path in required_files.items() if not path.is_file()
    ]
    if missing:
        raise ArtifactError(
            "missing",
            "Missing cached validation output: "
            + "; ".join(missing)
            + ". Run `python -m sunreserve validate` after preparing local cached data.",
        )

    manifest = _read_json(required_files["selection manifest"], "Selection manifest")
    metrics_payload = _read_json(required_files["validation metrics"], "Validation metrics")
    if (
        manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("artifact_type") != "sunreserve_validation_selection"
    ):
        raise ArtifactError(
            "incompatible", "Selection manifest schema or artifact type is unsupported."
        )
    if metrics_payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("incompatible", "Validation metrics schema version is unsupported.")
    if not str(manifest.get("status", "")).startswith("validation_only"):
        raise ArtifactError(
            "incompatible", "Only validation-only artifacts can be shown in this demo."
        )
    expected_output_hashes = manifest.get("development_artifact_hashes", {})
    expected_output_names = {
        "validation_predictions.csv",
        "validation_metrics.json",
        "decision_predictions.csv",
        "decision_metrics.csv",
    }
    if set(expected_output_hashes) != expected_output_names:
        raise ArtifactError("incompatible", "Manifest lacks required development artifact hashes.")
    for name, expected_hash in expected_output_hashes.items():
        try:
            actual_hash = hashlib.sha256((paths.validation_dir / name).read_bytes()).hexdigest()
        except OSError as error:
            raise ArtifactError("corrupt", f"Validation artifact {name} cannot be read.") from error
        if actual_hash != expected_hash:
            raise ArtifactError("stale", f"Validation artifact checksum does not match: {name}.")

    try:
        config = ExperimentConfig.from_dict(manifest["configuration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(
            "incompatible", f"Saved experiment configuration is invalid: {error}"
        ) from error
    if manifest.get("configuration_sha256") != _canonical_hash(config.to_dict()):
        raise ArtifactError("stale", "Saved configuration checksum does not match the manifest.")
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

    expected_code_hashes = _current_code_hashes((code_root or root).resolve())
    if manifest.get("code_hashes") != expected_code_hashes:
        raise ArtifactError(
            "stale", "Validation artifacts were created with different experiment code."
        )
    daily = _load_development_daily(paths, config)
    expected_data_hash = _development_data_hash(daily)
    input_hashes = manifest.get("input_hashes", {})
    quality = _read_json(paths.quality_path, "data quality report")
    sidecar = _read_json(paths.sidecar_path, "source provenance sidecar")
    if input_hashes.get("prepared_development_rows_sha256") != expected_data_hash:
        raise ArtifactError("stale", "Prepared validation inputs have changed since the run.")
    if input_hashes.get("raw_response_sha256") != sidecar.get("sha256"):
        raise ArtifactError(
            "stale", "Validation artifacts reference a different raw source response."
        )
    if (
        input_hashes.get("quality_report_sha256")
        != hashlib.sha256(paths.quality_path.read_bytes()).hexdigest()
    ):
        raise ArtifactError("stale", "Quality report changed after validation was frozen.")

    forecasts = _read_csv(required_files["validation predictions"], "Validation predictions")
    decisions = _read_csv(required_files["decision predictions"], "Decision predictions")
    decision_metrics = _read_csv(required_files["decision metrics"], "Decision metrics")
    _require_columns(forecasts, FORECAST_COLUMNS, "Validation predictions")
    _require_columns(decisions, DECISION_COLUMNS, "Decision predictions")
    _require_columns(decision_metrics, DECISION_METRIC_COLUMNS, "Decision metrics")
    if forecasts.empty or decisions.empty or decision_metrics.empty:
        raise ArtifactError("incompatible", "Validation outputs contain no rows.")

    for frame, label in (
        (forecasts, "Validation predictions"),
        (decisions, "Decision predictions"),
    ):
        try:
            frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
        except (TypeError, ValueError) as error:
            raise ArtifactError("corrupt", f"{label} contains an invalid target date.") from error
        if frame["target_date"].isna().any():
            raise ArtifactError("corrupt", f"{label} contains a missing target date.")
    try:
        forecasts["issue_date"] = pd.to_datetime(forecasts["issue_date"], errors="raise")
    except (TypeError, ValueError) as error:
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
    if (forecasts["issue_date"] >= forecasts["target_date"]).any():
        raise ArtifactError(
            "incompatible", "Validation predictions contain an issue date after the target."
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

    selected_model = manifest.get("model_selection", {}).get("selected_model")
    if selected_model not in PREDICTION_COLUMNS:
        raise ArtifactError("incompatible", "Manifest names an unsupported selected model.")
    if metrics_payload.get("selected_model") != selected_model:
        raise ArtifactError("stale", "Validation metrics and selection manifest disagree.")
    forecast_metrics = manifest.get("model_selection", {}).get("validation_metrics", {})
    if set(forecast_metrics) != set(PREDICTION_COLUMNS):
        raise ArtifactError(
            "incompatible", "Forecast metrics must include both baselines and Ridge."
        )

    try:
        decisions["approach"] = decisions["approach"].astype(str)
        decision_metrics["approach"] = decision_metrics["approach"].astype(str)
        scenarios = set(decisions["scenario_id"].astype(str))
        approaches = set(decisions["approach"])
        decision_dates = pd.DatetimeIndex(decisions["target_date"])
    except (TypeError, ValueError) as error:
        raise ArtifactError("corrupt", "Decision output keys are invalid.") from error
    if approaches != set(APPROACHES):
        raise ArtifactError(
            "incompatible", "Decision outputs must contain all three models and backup reference."
        )
    if decision_dates.has_duplicates is False and len(decisions) == 0:
        raise ArtifactError("incompatible", "Decision output contains no date rows.")
    expected_scenarios = 1 + int(
        manifest.get("decision_simulation", {}).get("sensitivity_grid", {}).get("scenario_count", 0)
    )
    if len(scenarios) != expected_scenarios:
        raise ArtifactError("incompatible", "Decision scenario count does not match the manifest.")
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

    numeric_decision_columns = tuple(DECISION_COLUMNS - {"scenario_id", "approach", "target_date"})
    for approach in APPROACHES:
        selected = decisions.loc[decisions["approach"] == approach]
        fields = tuple(
            name
            for name in numeric_decision_columns
            if not (
                approach == "always_prepare_backup"
                and name in {"predicted_radiation_kwh_m2_day", "predicted_generation_proxy_kwh_day"}
            )
        )
        _finite_columns(selected, fields, "Decision predictions")
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
    _finite_columns(
        decision_metrics,
        tuple(DECISION_METRIC_COLUMNS - {"scenario_id", "approach", "selected_model"}),
        "Decision metrics",
    )

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
