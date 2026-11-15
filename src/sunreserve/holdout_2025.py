"""Frozen, single-use evaluation on the separate 2025 historical holdout snapshot.

This module is intentionally separate from the validation/finalization code whose source
hashes are bound into the frozen development manifest. It never fits or selects a model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import DataError, ProjectPaths, parse_power_payload, request_url
from sunreserve.decision import PREDICTION_COLUMNS, simulate_daily_reserve
from sunreserve.evaluate import (
    _load_full_daily,
    _safe_run_id,
    _sha256_daily_frame,
    _sha256_file,
    _validate_frozen_development,
    forecast_metrics,
)
from sunreserve.features import RADIATION, build_next_day_features
from sunreserve.inference import load_model_bundle, predict_next_day

HOLDOUT_START = date(2025, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
HOLDOUT_SCHEMA_VERSION = 1
HOLDOUT_ARTIFACT_TYPE = "sunreserve_untouched_2025_holdout"
FROZEN_SELECTION_SHA256 = "9cd60c7ab8b4e41078c1bf44d6698178334e6b9b6dba3d01a88f4e9a9d097fff"
FROZEN_MODEL_SHA256 = "40a11e5119b7838d10a4a7ef7cc5bbf564998759b3954d4b865fdfe3f392b61f"
DECISION_SCORE_NAME = "mean_shortfall_plus_mean_excess_kwh_day"
EXPECTED_DECISION_OBJECTIVE = (
    "mean simulated shortfall kWh/day + mean excess prepared reserve kWh/day; "
    "equal 1:1 energy weights, not monetary costs"
)
HOLDOUT_APPROACHES = ("ridge", "persistence", "seasonal_mean")


@dataclass(frozen=True)
class HoldoutRun:
    """Saved path and common-target count for a completed 2025 evaluation."""

    run_dir: Path
    sample_count: int
    model_id: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise DataError(f"2025 provenance JSON contains duplicate key {key!r}")
        parsed[key] = value
    return parsed


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_no_duplicate_keys
        )
    except (OSError, json.JSONDecodeError, DataError) as error:
        raise DataError(f"{label} is missing, unreadable, or invalid: {error}") from error
    if not isinstance(parsed, dict):
        raise DataError(f"{label} must contain a JSON object")
    return parsed


def _holdout_paths(root: Path) -> tuple[Path, Path]:
    stem = f"power_daily_{root.name}_2025_holdout"
    raw = root / "data" / "raw" / f"{stem}.json"
    return raw, raw.with_suffix(".meta.json")


def _holdout_config(config: ExperimentConfig) -> ExperimentConfig:
    return replace(config, start_date=HOLDOUT_START, end_date=HOLDOUT_END)


def _load_holdout_daily(
    root: Path, config: ExperimentConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path, provenance_path = _holdout_paths(root)
    if not raw_path.is_file() or not provenance_path.is_file():
        raise DataError(
            "the separate 2025 holdout snapshot or its provenance sidecar is missing; "
            f"expected {raw_path} and {provenance_path}"
        )
    raw = raw_path.read_bytes()
    provenance = _read_object(provenance_path, "2025 source provenance sidecar")
    if provenance.get("schema_version") != 2 or provenance.get("source") != config.source:
        raise DataError("2025 provenance schema version or source is incompatible")
    if provenance.get("missing_value_marker") != -999.0:
        raise DataError("2025 provenance missing-value marker is incompatible")
    digest = _sha256_bytes(raw)
    if provenance.get("sha256") != digest:
        raise DataError("2025 source snapshot checksum does not match its provenance sidecar")
    holdout_config = _holdout_config(config)
    if provenance.get("request_url") != request_url(holdout_config):
        raise DataError(
            "2025 source request does not match the frozen location, parameters, or period"
        )
    request_parameters = provenance.get("request_parameters")
    expected_parameters = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "start_date": HOLDOUT_START.isoformat(),
        "end_date": HOLDOUT_END.isoformat(),
        "parameters": list(config.parameters),
        "community": config.community,
        "time_standard": "UTC",
        "format": "JSON",
    }
    if request_parameters != expected_parameters:
        raise DataError("2025 provenance request parameters are incomplete or incompatible")
    retrieved = provenance.get("retrieved_utc")
    try:
        retrieved_at = datetime.fromisoformat(retrieved)
    except (TypeError, ValueError) as error:
        raise DataError("2025 provenance retrieved_utc must be an ISO timestamp") from error
    if retrieved_at.utcoffset() is None or retrieved_at.utcoffset().total_seconds() != 0:
        raise DataError("2025 provenance retrieved_utc must use UTC")

    try:
        frame, source_metadata = parse_power_payload(raw, holdout_config)
    except (DataError, ValueError) as error:
        raise DataError(
            f"2025 source snapshot failed schema, unit, date, or location validation: {error}"
        ) from error
    expected_dates = pd.date_range(HOLDOUT_START, HOLDOUT_END, freq="D")
    if not frame.index.equals(expected_dates):
        raise DataError("2025 source parser did not produce the requested calendar")
    observed_units = {
        name: metadata.get("units")
        for name, metadata in source_metadata["parameters"].items()
        if name in config.parameters
    }
    if provenance.get("units") != observed_units:
        raise DataError("2025 provenance units do not match the validated NASA POWER response")
    reported_location = source_metadata["reported_location"]
    if provenance.get("reported_location") != reported_location:
        raise DataError("2025 provenance location does not match the NASA POWER response")
    missing_by_variable = {name: int(frame[name].isna().sum()) for name in config.parameters}
    quality = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "source": config.source,
        "source_period": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "time_standard": "UTC",
        "requested_location": {
            "latitude": config.latitude,
            "longitude": config.longitude,
        },
        "reported_location": reported_location,
        "units": observed_units,
        "raw_sha256": digest,
        "retrieved_utc": retrieved,
        "calendar_days": len(frame),
        "missing_counts_by_variable": missing_by_variable,
        "complete_case_days": int(frame.loc[:, list(config.parameters)].notna().all(axis=1).sum()),
    }
    return frame, {
        "quality": quality,
        "provenance": provenance,
        "raw_path": raw_path,
        "provenance_path": provenance_path,
    }


def _validate_frozen_decision_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    decision = manifest.get("decision_selection")
    if not isinstance(decision, dict):
        raise DataError("frozen development manifest has no decision-selection contract")
    if (
        decision.get("objective") != EXPECTED_DECISION_OBJECTIVE
        or decision.get("score_name") != DECISION_SCORE_NAME
        or decision.get("shortfall_weight") != 1.0
        or decision.get("excess_prepared_reserve_weight") != 1.0
        or decision.get("selection_partition") != "validation"
        or decision.get("scenario_id") != "primary"
    ):
        raise DataError("frozen decision objective or selection settings are incompatible")
    if decision.get("selection_target_period") != ["2022-01-01", "2022-12-31"]:
        raise DataError("decision model must have been selected on the 2022 validation period")
    if decision.get("selected_model") not in HOLDOUT_APPROACHES:
        raise DataError("frozen decision choice is not an available holdout approach")
    return decision


def _build_common_predictions(
    previous_daily: pd.DataFrame,
    holdout_daily: pd.DataFrame,
    model: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Predict 2025 targets only, using earlier dates solely as issue-window context."""
    required = {RADIATION, "T2M", "RH2M"}
    if not required <= set(previous_daily.columns) or not required <= set(holdout_daily.columns):
        raise DataError("historical context or holdout source is missing required daily fields")
    if not previous_daily.index.is_monotonic_increasing or previous_daily.index.has_duplicates:
        raise DataError("historical context must have sorted unique dates")
    if not holdout_daily.index.is_monotonic_increasing or holdout_daily.index.has_duplicates:
        raise DataError("2025 holdout rows must have sorted unique dates")
    if holdout_daily.index.min() != pd.Timestamp(
        HOLDOUT_START
    ) or holdout_daily.index.max() != pd.Timestamp(HOLDOUT_END):
        raise DataError("2025 holdout calendar is not bounded to the frozen year")
    combined = pd.concat(
        [previous_daily.loc[:, list(required)], holdout_daily.loc[:, list(required)]]
    ).sort_index()
    if combined.index.has_duplicates:
        raise DataError("historical context and 2025 holdout dates overlap")
    target_dates = pd.date_range(HOLDOUT_START, HOLDOUT_END, freq="D")
    predictions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for target_date in target_dates:
        issue_date = target_date - pd.Timedelta(days=1)
        start_date = issue_date - pd.Timedelta(days=6)
        target_value = holdout_daily.at[target_date, RADIATION]
        if pd.isna(target_value) or not np.isfinite(float(target_value)):
            excluded.append(
                {
                    "target_date": target_date.date().isoformat(),
                    "reason": "missing target radiation",
                }
            )
            continue
        expected_window = pd.date_range(start_date, issue_date, freq="D")
        window = combined.reindex(expected_window)
        if len(window) != 7 or window.isna().to_numpy().any():
            excluded.append(
                {
                    "target_date": target_date.date().isoformat(),
                    "reason": "missing or incomplete seven-day issue-date feature window",
                }
            )
            continue
        window.index.name = "date"
        try:
            build_next_day_features(window)
            result = predict_next_day(model, window)
        except ValueError as error:
            excluded.append(
                {
                    "target_date": target_date.date().isoformat(),
                    "reason": f"invalid feature window: {error}",
                }
            )
            continue
        if result.target_date != target_date.date():
            raise RuntimeError("inference target alignment is not issue_date + one calendar day")
        if result.issue_date != issue_date.date():
            raise RuntimeError("inference issue date does not match the holdout target date")
        predictions.append(
            {
                "issue_date": issue_date.date().isoformat(),
                "target_date": target_date.date().isoformat(),
                "split": "untouched_2025_holdout",
                "target_radiation_kwh_m2_day": float(target_value),
                "persistence_prediction_kwh_m2_day": result.persistence_kwh_m2_day,
                "seasonal_mean_prediction_kwh_m2_day": result.seasonal_mean_kwh_m2_day,
                "ridge_prediction_kwh_m2_day": result.radiation_kwh_m2_day,
            }
        )
    result_frame = pd.DataFrame(predictions)
    if result_frame.empty:
        raise DataError(
            "no eligible 2025 target dates remain after input/target missingness exclusions"
        )
    result_frame["issue_date"] = pd.to_datetime(result_frame["issue_date"])
    result_frame["target_date"] = pd.to_datetime(result_frame["target_date"])
    expected_issue = result_frame["target_date"] - pd.Timedelta(days=1)
    if not result_frame["issue_date"].equals(expected_issue):
        raise RuntimeError("2025 prediction rows do not align issue dates and target dates")
    if not result_frame["target_date"].dt.year.eq(2025).all():
        raise RuntimeError("holdout predictions contain targets outside 2025")
    if result_frame["target_date"].duplicated().any():
        raise RuntimeError("holdout predictions contain duplicate target dates")
    prediction_columns = list(PREDICTION_COLUMNS.values())
    if (
        not np.isfinite(result_frame[["target_radiation_kwh_m2_day", *prediction_columns]])
        .all()
        .all()
    ):
        raise RuntimeError("common holdout forecasts or targets contain nonfinite values")
    return result_frame, excluded


def _forecast_metrics(predictions: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    threshold = manifest["model_selection"]["training_only_low_solar_threshold_kwh_m2_day"]
    actual = predictions["target_radiation_kwh_m2_day"].to_numpy(dtype=float)
    dates = predictions["target_date"]
    metric_values = {}
    for approach, column in PREDICTION_COLUMNS.items():
        metric_values[approach] = forecast_metrics(
            actual,
            predictions[column].to_numpy(dtype=float),
            dates,
            low_solar_threshold=float(threshold),
        )
    return {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "partition": "untouched_2025_holdout",
        "target_period": [
            predictions["target_date"].min().date().isoformat(),
            predictions["target_date"].max().date().isoformat(),
        ],
        "sample_count": int(len(predictions)),
        "target_units": "kWh/m^2/day",
        "low_solar_threshold_frozen_from_training_kwh_m2_day": float(threshold),
        "metrics": metric_values,
    }


def _decision_outputs(
    predictions: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario = {
        "panel_area_m2": config.panel_area_m2,
        "efficiency": config.conversion_efficiency,
        "derating": config.derating_factor,
        "demand_kwh": config.daily_demand_kwh,
        "capacity_kwh": config.reserve_cap_kwh,
        "buffer_kwh": config.reserve_buffer_kwh,
    }
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    actual = predictions["target_radiation_kwh_m2_day"].to_numpy(dtype=float)
    for approach, column in PREDICTION_COLUMNS.items():
        decision = simulate_daily_reserve(
            actual,
            predictions[column].to_numpy(dtype=float),
            demand_kwh=scenario["demand_kwh"],
            capacity_kwh=scenario["capacity_kwh"],
            buffer_kwh=scenario["buffer_kwh"],
            panel_area_m2=scenario["panel_area_m2"],
            efficiency=scenario["efficiency"],
            derating=scenario["derating"],
        )
        frame = decision.assign(
            issue_date=predictions["issue_date"].to_numpy(),
            target_date=predictions["target_date"].to_numpy(),
            scenario_id="primary",
            approach=approach,
            actual_radiation_kwh_m2_day=actual,
            predicted_radiation_kwh_m2_day=predictions[column].to_numpy(dtype=float),
        )
        frame = frame[
            [
                "scenario_id",
                "approach",
                "issue_date",
                "target_date",
                "actual_radiation_kwh_m2_day",
                "predicted_radiation_kwh_m2_day",
                *decision.columns,
            ]
        ]
        shortfall = decision["simulated_shortfall_kwh"].to_numpy(dtype=float)
        excess = decision["excess_prepared_reserve_kwh"].to_numpy(dtype=float)
        summaries.append(
            {
                "scenario_id": "primary",
                "approach": approach,
                "sample_count": int(len(decision)),
                "total_simulated_shortfall_kwh": float(shortfall.sum()),
                "mean_simulated_shortfall_kwh_day": float(shortfall.mean()),
                "simulated_shortfall_day_count": int(np.count_nonzero(shortfall > 0)),
                "simulated_shortfall_day_share": float(np.mean(shortfall > 0)),
                "total_excess_prepared_reserve_kwh": float(excess.sum()),
                "mean_excess_prepared_reserve_kwh_day": float(excess.mean()),
                "decision_score_name": DECISION_SCORE_NAME,
                "mean_shortfall_plus_mean_excess_kwh_day": float(shortfall.mean() + excess.mean()),
                "shortfall_weight": 1.0,
                "excess_prepared_reserve_weight": 1.0,
                "total_capacity_limited_shortfall_kwh": float(
                    decision["capacity_limited_shortfall_kwh"].sum()
                ),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def _render_report(
    *,
    run_id: str,
    manifest: dict[str, Any],
    model_id: str,
    sample_count: int,
    excluded_count: int,
    forecast: dict[str, Any],
    decisions: pd.DataFrame,
) -> str:
    rows = []
    for approach in HOLDOUT_APPROACHES:
        values = forecast["metrics"][approach]
        decision = decisions.loc[decisions["approach"] == approach].iloc[0]
        rows.append(
            f"| {approach} | {values['mae']:.4f} | {values['rmse']:.4f} | "
            f"{decision['mean_simulated_shortfall_kwh_day']:.4f} | "
            f"{decision['mean_excess_prepared_reserve_kwh_day']:.4f} | "
            f"{decision['mean_shortfall_plus_mean_excess_kwh_day']:.4f} |"
        )
    decision = manifest["decision_selection"]
    return (
        f"# SunReserve untouched 2025 holdout · {run_id}\n\n"
        "This run applies the frozen model and validation-selected decision contract to the "
        "separate 2025 historical period. It does not refit, tune, or select on 2025. "
        "Targets cover 2025 only; 2023–2024 are not scored in this run.\n\n"
        f"Common eligible targets: {sample_count}; excluded 2025 target dates: {excluded_count}. "
        f"Frozen model: `{model_id}`.\n\n"
        "Radiation metrics use `kWh/m²/day`. Reserve outcomes use hypothetical simulated "
        "`kWh/day` assumptions and are not real facility performance. The decision score is "
        f"`{decision['objective']}`. The development-selected approach was "
        f"`{decision['selected_model']}` using 2022 validation only.\n\n"
        "| Approach | Radiation MAE | Radiation RMSE | Mean shortfall (kWh/day) | "
        "Mean excess reserve (kWh/day) | Mean shortfall + excess (kWh/day) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(rows) + "\n\n"
        "These are historical estimates. NASA POWER publication latency and user-input "
        "provenance do not establish forecast-time availability. The electricity conversion, "
        "demand, capacity, and buffer remain hypothetical; this is not an operational "
        "controller, cost model, or causal analysis. See `docs/assumptions.md` and "
        "`docs/data_sources.md`.\n"
    )


def evaluate_2025_holdout(
    root: Path,
    *,
    manifest_sha256: str,
    model_sha256: str,
    run_id: str,
) -> HoldoutRun:
    """Evaluate frozen candidates on 2025 once; refuse mismatches and overwrite."""
    project_root = root.resolve()
    paths = ProjectPaths(project_root)
    safe_run_id = _safe_run_id(run_id)
    run_dir = paths.runs_dir / safe_run_id
    if run_dir.exists():
        raise DataError(f"holdout run already exists and will not be overwritten: {run_dir}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", model_sha256):
        raise ValueError("--model-sha256 must be a 64-character SHA-256 hex digest")
    if manifest_sha256.lower() != FROZEN_SELECTION_SHA256:
        raise DataError("selection manifest digest is not the explicitly frozen review digest")
    if model_sha256.lower() != FROZEN_MODEL_SHA256:
        raise DataError("model digest is not the explicitly frozen review bundle digest")

    manifest, config, _, _ = _validate_frozen_development(paths, manifest_sha256, None)
    bundle_manifest = manifest.get("model_bundle", {})
    if bundle_manifest.get("sha256", "").lower() != model_sha256.lower():
        raise DataError("model SHA-256 does not match the frozen validation manifest")
    if _sha256_file(paths.model_path).lower() != model_sha256.lower():
        raise DataError("trained model bundle does not match the supplied frozen SHA-256")
    selection = _validate_frozen_decision_contract(manifest)
    model = load_model_bundle(paths.model_path, expected_sha256=model_sha256.lower())
    if model.model_id != bundle_manifest.get("model_id"):
        raise DataError("trained model identity does not match the frozen selection manifest")

    # Context through 2024 is used only to construct 2025 issue-date features. No 2023–2024
    # target rows are scored or passed to model/policy selection.
    previous_daily = _load_full_daily(paths, config)
    holdout_daily, source_info = _load_holdout_daily(project_root, config)
    predictions, excluded = _build_common_predictions(previous_daily, holdout_daily, model)
    expected_dates = pd.DatetimeIndex(predictions["target_date"])
    if (expected_dates.year != 2025).any():
        raise RuntimeError("refusing to write a run containing non-2025 targets")
    if predictions["target_date"].duplicated().any():
        raise RuntimeError("holdout predictions do not use unique target dates")

    forecast = _forecast_metrics(predictions, manifest)
    decision_predictions, decision_metrics = _decision_outputs(predictions, config)
    if set(decision_metrics["approach"]) != set(HOLDOUT_APPROACHES):
        raise RuntimeError("decision metrics omit a frozen candidate approach")
    if set(decision_metrics["sample_count"].astype(int)) != {len(predictions)}:
        raise RuntimeError("candidate reserve metrics do not use the same 2025 target dates")
    holdout_config = _holdout_config(config)
    selection_manifest_bytes = (paths.validation_dir / "selection_manifest.json").read_bytes()
    code_paths = {
        "holdout_2025": Path(__file__).resolve(),
        "holdout_2025_cli": Path(__file__).with_name("holdout_2025_cli.py").resolve(),
    }
    code_hashes = {name: _sha256_file(path) for name, path in code_paths.items()}
    prediction_content = predictions.to_csv(index=False, lineterminator="\n")
    decision_prediction_content = decision_predictions.to_csv(index=False, lineterminator="\n")
    decision_metrics_content = decision_metrics.to_csv(index=False, lineterminator="\n")
    forecast_content = json.dumps(forecast, indent=2, sort_keys=True, allow_nan=False) + "\n"
    quality = source_info["quality"] | {
        "eligible_targets": int(len(predictions)),
        "excluded_target_count": len(excluded),
        "excluded_targets": excluded,
        "context_period_used_for_lags_only": [
            previous_daily.index.min().date().isoformat(),
            previous_daily.index.max().date().isoformat(),
        ],
    }
    quality_content = json.dumps(quality, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_content = _render_report(
        run_id=safe_run_id,
        manifest=manifest,
        model_id=model.model_id,
        sample_count=len(predictions),
        excluded_count=len(excluded),
        forecast=forecast,
        decisions=decision_metrics,
    )
    content_files = {
        "predictions.csv": prediction_content,
        "forecast_metrics.json": forecast_content,
        "decision_predictions.csv": decision_prediction_content,
        "decision_metrics.csv": decision_metrics_content,
        "data_quality.json": quality_content,
        "selection_manifest.json": selection_manifest_bytes.decode("utf-8"),
        "report.md": report_content,
    }
    artifact_hashes = {
        name: _sha256_bytes(value.encode("utf-8")) for name, value in content_files.items()
    }
    holdout_manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "artifact_type": HOLDOUT_ARTIFACT_TYPE,
        "run_id": safe_run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "partition": "untouched_2025_holdout",
        "target_period": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
        "selection_manifest_sha256": manifest_sha256.lower(),
        "model_bundle_sha256": model_sha256.lower(),
        "model_id": model.model_id,
        "model_training_split": "train",
        "model_training_rows": model.training_rows,
        "model_training_target_period": [
            model.training_start.isoformat(),
            model.training_end.isoformat(),
        ],
        "decision_selection": selection,
        "decision_scenario": manifest["decision_simulation"]["primary_scenario"],
        "holdout_configuration": holdout_config.to_dict(),
        "holdout_configuration_sha256": _canonical_hash(holdout_config.to_dict()),
        "historical_context_sha256": _sha256_daily_frame(previous_daily),
        "source_snapshot": {
            "path": source_info["raw_path"].relative_to(project_root).as_posix(),
            "sha256": source_info["quality"]["raw_sha256"],
            "provenance_path": source_info["provenance_path"].relative_to(project_root).as_posix(),
            "provenance_sha256": _sha256_file(source_info["provenance_path"]),
            "request_url": source_info["provenance"]["request_url"],
            "retrieved_utc": source_info["provenance"]["retrieved_utc"],
        },
        "code_revision": "uncommitted",
        "frozen_development_code_hashes": manifest["code_hashes"],
        "holdout_evaluation_code_hashes": code_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "row_counts": {
            "eligible_2025_targets": int(len(predictions)),
            "excluded_2025_targets": len(excluded),
            "2023_2024_targets_scored": 0,
        },
        "artifact_schemas": {
            name: int(manifest["schema_version"])
            if name == "selection_manifest.json"
            else HOLDOUT_SCHEMA_VERSION
            for name in content_files
        },
        "artifact_sha256": artifact_hashes,
        "results_are_descriptive": True,
        "no_2025_tuning_or_selection": True,
    }
    content_files["run_manifest.json"] = (
        json.dumps(holdout_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )

    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise DataError(f"holdout run already exists and will not be overwritten: {run_dir}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{safe_run_id}.", dir=paths.runs_dir))
    try:
        for name, content in content_files.items():
            (temporary_dir / name).write_text(content, encoding="utf-8", newline="\n")
        temporary_dir.replace(run_dir)
    except OSError:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return HoldoutRun(run_dir=run_dir, sample_count=len(predictions), model_id=model.model_id)
