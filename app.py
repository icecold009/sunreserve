"""Fresh trained-model inference and cached historical replay for SunReserve."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sunreserve.config import ExperimentConfig
from sunreserve.data import ProjectPaths
from sunreserve.decision import APPROACHES, PREDICTION_COLUMNS, RADIATION_COLUMN, plan_daily_reserve
from sunreserve.features import RADIATION
from sunreserve.inference import (
    ModelArtifactError,
    load_project_model,
    observation_frame_from_records,
    predict_next_day,
)
from sunreserve.observations import OBSERVATION_POLICY_NOTE
from sunreserve.prediction_export import (
    build_prediction_export,
    entrypoint_source_identity,
    generated_at_utc_now,
)
from sunreserve.presentation import (
    ArtifactError,
    DemoArtifacts,
    artifact_signature,
    failure_examples,
    list_validation_experiments,
    load_demo_artifacts,
)

APPROACH_LABELS = {
    "persistence": "Persistence baseline",
    "seasonal_mean": "Training-only seasonal mean baseline",
    "ridge": "Ridge regression",
    "always_prepare_backup": "Always prepare backup reference",
}
MOBILE_METRIC_VALUE_MAX_CHARS = 13
PREDICTION_STATE_KEY = "sunreserve_last_prediction"
OBSERVATION_EDITOR_KEY = "sunreserve_prediction_observations"
OBSERVATION_EDITOR_DATA_KEY = "sunreserve_prediction_observations_data"
OBSERVATION_EDITOR_VERSION_KEY = "sunreserve_prediction_observations_version"
CSV_UPLOAD_KEY = "sunreserve_csv_upload"
CSV_UPLOAD_VERSION_KEY = "sunreserve_csv_upload_version"
CSV_UPLOAD_PROCESSED_KEY = "sunreserve_csv_upload_processed_sha256"
CSV_UPLOAD_STATUS_KEY = "sunreserve_csv_upload_status"
GUIDED_EXAMPLE_KEY = "sunreserve_guided_example_applied"
OBSERVATION_ORIGIN_KEY = "sunreserve_prediction_observation_origin"
OBSERVATION_COLUMNS = ["date", RADIATION, "T2M", "RH2M"]
FROZEN_2025_FILES = ("predictions.csv", "decision_metrics.csv", "forecast_metrics.json")
FROZEN_2025_PREDICTION_COLUMNS = {
    "issue_date",
    "target_date",
    "split",
    "target_radiation_kwh_m2_day",
    "persistence_prediction_kwh_m2_day",
    "seasonal_mean_prediction_kwh_m2_day",
    "ridge_prediction_kwh_m2_day",
}
FROZEN_2025_DECISION_COLUMNS = {
    "scenario_id",
    "approach",
    "sample_count",
    "total_simulated_shortfall_kwh",
    "mean_simulated_shortfall_kwh_day",
    "simulated_shortfall_day_count",
    "simulated_shortfall_day_share",
    "total_excess_prepared_reserve_kwh",
    "mean_excess_prepared_reserve_kwh_day",
    "decision_score_name",
    "mean_shortfall_plus_mean_excess_kwh_day",
    "shortfall_weight",
    "excess_prepared_reserve_weight",
    "total_capacity_limited_shortfall_kwh",
}
FROZEN_2025_FORECAST_METRIC_FIELDS = {
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


class Frozen2025ArtifactsError(RuntimeError):
    """Raised when the committed frozen 2025 result artifacts cannot be verified."""


def _frozen_json_object(payload: bytes, label: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} contains a non-finite number")
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Frozen2025ArtifactsError(f"{label} cannot be read: {error}") from error
    if not isinstance(value, dict):
        raise Frozen2025ArtifactsError(f"{label} must contain a JSON object.")
    return value


def _frozen_number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Frozen2025ArtifactsError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except OverflowError as error:
        raise Frozen2025ArtifactsError(f"{label} must be finite.") from error
    if not math.isfinite(number):
        raise Frozen2025ArtifactsError(f"{label} must be finite.")
    if nonnegative and number < 0:
        raise Frozen2025ArtifactsError(f"{label} must be nonnegative.")
    return number


def _require_frozen_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    if frame.columns.has_duplicates:
        raise Frozen2025ArtifactsError(f"{label} contains duplicate columns.")
    missing = expected - set(frame.columns)
    extra = set(frame.columns) - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unsupported {', '.join(sorted(extra))}")
        raise Frozen2025ArtifactsError(f"{label} has an invalid schema: {'; '.join(details)}.")


def _validate_frozen_forecast_metrics(
    payload: dict[str, object], predictions: pd.DataFrame, expected_dates: pd.DatetimeIndex
) -> None:
    expected_top_fields = {
        "schema_version",
        "partition",
        "target_period",
        "target_units",
        "sample_count",
        "low_solar_threshold_frozen_from_training_kwh_m2_day",
        "metrics",
    }
    if set(payload) != expected_top_fields:
        raise Frozen2025ArtifactsError("Frozen forecast metrics have an unsupported schema.")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise Frozen2025ArtifactsError("Frozen forecast metrics schema version is unsupported.")
    if (
        payload["partition"] != "untouched_2025_holdout"
        or payload["target_period"] != ["2025-01-01", "2025-12-31"]
        or payload["target_units"] != "kWh/m^2/day"
    ):
        raise Frozen2025ArtifactsError(
            "Frozen forecast metrics partition, period, or units are invalid."
        )
    if type(payload["sample_count"]) is not int or payload["sample_count"] != len(expected_dates):
        raise Frozen2025ArtifactsError("Frozen forecast metrics have an invalid sample count.")
    threshold = _frozen_number(
        payload["low_solar_threshold_frozen_from_training_kwh_m2_day"],
        "Frozen low-solar threshold",
        nonnegative=True,
    )
    metrics = payload["metrics"]
    expected_approaches = {"persistence", "seasonal_mean", "ridge"}
    if not isinstance(metrics, dict) or set(metrics) != expected_approaches:
        raise Frozen2025ArtifactsError("Frozen forecast metrics must include all supported models.")

    expected_month_counts = {
        f"{month:02d}": int((expected_dates.month == month).sum()) for month in range(1, 13)
    }
    season_months = {
        "DJF": {12, 1, 2},
        "MAM": {3, 4, 5},
        "JJA": {6, 7, 8},
        "SON": {9, 10, 11},
    }
    expected_season_counts = {
        season: int(expected_dates.month.isin(months).sum())
        for season, months in season_months.items()
    }
    columns = {
        "persistence": "persistence_prediction_kwh_m2_day",
        "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
        "ridge": "ridge_prediction_kwh_m2_day",
    }
    actual = predictions["target_radiation_kwh_m2_day"].to_numpy(dtype=float)
    for approach, prediction_column in columns.items():
        metric = metrics[approach]
        if not isinstance(metric, dict) or set(metric) != FROZEN_2025_FORECAST_METRIC_FIELDS:
            raise Frozen2025ArtifactsError(f"Frozen {approach} metric schema is invalid.")
        if type(metric["sample_count"]) is not int or metric["sample_count"] != len(expected_dates):
            raise Frozen2025ArtifactsError(
                f"Frozen {approach} metrics have an invalid sample count."
            )
        mae = _frozen_number(metric["mae"], f"Frozen {approach} MAE", nonnegative=True)
        rmse = _frozen_number(metric["rmse"], f"Frozen {approach} RMSE", nonnegative=True)
        bias = _frozen_number(metric["bias_prediction_minus_actual"], f"Frozen {approach} bias")
        if (
            metric["prediction_clip_rule"]
            != "max(0, prediction), applied to every candidate before scoring"
        ):
            raise Frozen2025ArtifactsError(
                f"Frozen {approach} prediction clipping rule is invalid."
            )

        monthly_mae = metric["monthly_mae_by_target_calendar_month"]
        monthly_counts = metric["monthly_sample_count"]
        if (
            not isinstance(monthly_mae, dict)
            or set(monthly_mae) != set(expected_month_counts)
            or not isinstance(monthly_counts, dict)
            or monthly_counts != expected_month_counts
        ):
            raise Frozen2025ArtifactsError(
                f"Frozen {approach} monthly metrics have an invalid schema."
            )
        for month, value in monthly_mae.items():
            _frozen_number(value, f"Frozen {approach} monthly MAE {month}", nonnegative=True)
        if any(type(count) is not int or count <= 0 for count in monthly_counts.values()):
            raise Frozen2025ArtifactsError(f"Frozen {approach} monthly sample counts are invalid.")

        seasonal_mae = metric["seasonal_mae"]
        seasonal_counts = metric["seasonal_sample_count"]
        if (
            not isinstance(seasonal_mae, dict)
            or set(seasonal_mae) != set(expected_season_counts)
            or not isinstance(seasonal_counts, dict)
            or seasonal_counts != expected_season_counts
        ):
            raise Frozen2025ArtifactsError(
                f"Frozen {approach} seasonal metrics have an invalid schema."
            )
        for season, value in seasonal_mae.items():
            _frozen_number(value, f"Frozen {approach} seasonal MAE {season}", nonnegative=True)
        if any(type(count) is not int or count <= 0 for count in seasonal_counts.values()):
            raise Frozen2025ArtifactsError(f"Frozen {approach} seasonal sample counts are invalid.")

        low_solar = metric["low_solar"]
        expected_low_fields = {
            "threshold_kwh_m2_day",
            "sample_count",
            "mae",
            "rmse",
            "bias_prediction_minus_actual",
        }
        if not isinstance(low_solar, dict) or set(low_solar) != expected_low_fields:
            raise Frozen2025ArtifactsError(f"Frozen {approach} low-solar metric schema is invalid.")
        low_threshold = _frozen_number(
            low_solar["threshold_kwh_m2_day"],
            f"Frozen {approach} low-solar threshold",
            nonnegative=True,
        )
        if not math.isclose(low_threshold, threshold, rel_tol=0, abs_tol=1e-12):
            raise Frozen2025ArtifactsError(
                f"Frozen {approach} low-solar threshold is inconsistent."
            )
        low_count = low_solar["sample_count"]
        if type(low_count) is not int or not 0 <= low_count <= len(expected_dates):
            raise Frozen2025ArtifactsError(f"Frozen {approach} low-solar sample count is invalid.")
        low_fields = ("mae", "rmse", "bias_prediction_minus_actual")
        if low_count == 0:
            if any(low_solar[name] is not None for name in low_fields):
                raise Frozen2025ArtifactsError(
                    f"Frozen {approach} empty low-solar metrics must be null."
                )
        else:
            _frozen_number(low_solar["mae"], f"Frozen {approach} low-solar MAE", nonnegative=True)
            _frozen_number(low_solar["rmse"], f"Frozen {approach} low-solar RMSE", nonnegative=True)
            _frozen_number(
                low_solar["bias_prediction_minus_actual"], f"Frozen {approach} low-solar bias"
            )

        prediction = predictions[prediction_column].to_numpy(dtype=float)
        error = prediction - actual
        recomputed = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias_prediction_minus_actual": float(np.mean(error)),
        }
        for field, expected_value in recomputed.items():
            actual_value = {"mae": mae, "rmse": rmse, "bias_prediction_minus_actual": bias}[field]
            if not math.isclose(actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-10):
                raise Frozen2025ArtifactsError(
                    f"Frozen {approach} {field} does not match predictions.csv."
                )


def _load_frozen_2025(project_root: Path) -> dict[str, object]:
    """Read the frozen result files and verify their manifest-bound checksums."""
    run_dir = project_root / "artifacts" / "runs" / "frozen-test-2025"
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        file_bytes = {name: (run_dir / name).read_bytes() for name in FROZEN_2025_FILES}
    except OSError as error:
        raise Frozen2025ArtifactsError(f"Frozen 2025 results are not available: {error}") from error
    manifest = _frozen_json_object(manifest_bytes, "Frozen run manifest")
    required_manifest_fields = {
        "schema_version",
        "artifact_type",
        "partition",
        "target_period",
        "artifact_schemas",
        "artifact_sha256",
        "row_counts",
        "decision_scenario",
        "decision_selection",
        "no_2025_tuning_or_selection",
        "results_are_descriptive",
    }
    if required_manifest_fields - set(manifest):
        missing = sorted(required_manifest_fields - set(manifest))
        raise Frozen2025ArtifactsError(
            f"Frozen run manifest is missing required fields: {', '.join(missing)}."
        )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise Frozen2025ArtifactsError("Frozen run manifest schema version is unsupported.")
    if (
        manifest["artifact_type"] != "sunreserve_untouched_2025_holdout"
        or manifest["partition"] != "untouched_2025_holdout"
        or manifest["target_period"] != ["2025-01-01", "2025-12-31"]
    ):
        raise Frozen2025ArtifactsError("Frozen run manifest identity or target period is invalid.")
    if (
        manifest["no_2025_tuning_or_selection"] is not True
        or manifest["results_are_descriptive"] is not True
    ):
        raise Frozen2025ArtifactsError("Frozen run manifest holdout declarations are invalid.")

    expected_schemas = {
        "predictions.csv": 1,
        "decision_metrics.csv": 1,
        "forecast_metrics.json": 1,
    }
    schemas = manifest["artifact_schemas"]
    if not isinstance(schemas, dict) or any(
        type(schemas.get(name)) is not int or schemas.get(name) != version
        for name, version in expected_schemas.items()
    ):
        raise Frozen2025ArtifactsError("Frozen output artifact schemas are unsupported.")

    row_counts = manifest["row_counts"]
    expected_row_counts = {
        "2023_2024_targets_scored": 0,
        "eligible_2025_targets": 365,
        "excluded_2025_targets": 0,
    }
    if (
        not isinstance(row_counts, dict)
        or set(row_counts) != set(expected_row_counts)
        or any(type(value) is not int for value in row_counts.values())
        or row_counts != expected_row_counts
    ):
        raise Frozen2025ArtifactsError("Frozen run manifest row counts are invalid.")

    checksums = manifest["artifact_sha256"]
    if not isinstance(checksums, dict):
        raise Frozen2025ArtifactsError("Frozen run manifest checksums must be an object.")
    for name, payload in file_bytes.items():
        expected = checksums.get(name)
        actual = hashlib.sha256(payload).hexdigest()
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
            or actual != expected.lower()
        ):
            raise Frozen2025ArtifactsError(f"Frozen 2025 artifact checksum mismatch: {name}")

    try:
        predictions = pd.read_csv(io.BytesIO(file_bytes["predictions.csv"]))
        decisions = pd.read_csv(io.BytesIO(file_bytes["decision_metrics.csv"]))
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        raise Frozen2025ArtifactsError(f"Frozen 2025 results cannot be read: {error}") from error
    forecast_metrics = _frozen_json_object(
        file_bytes["forecast_metrics.json"], "Frozen forecast metrics"
    )

    try:
        _require_frozen_columns(
            predictions, FROZEN_2025_PREDICTION_COLUMNS, "Frozen 2025 predictions"
        )
        _require_frozen_columns(
            decisions, FROZEN_2025_DECISION_COLUMNS, "Frozen 2025 decision metrics"
        )
        if predictions.empty or decisions.empty:
            raise Frozen2025ArtifactsError("Frozen 2025 outputs contain no rows.")
        predictions["issue_date"] = pd.to_datetime(predictions["issue_date"], errors="raise")
        predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="raise")
    except (OverflowError, TypeError, ValueError) as error:
        if isinstance(error, Frozen2025ArtifactsError):
            raise
        raise Frozen2025ArtifactsError(f"Frozen 2025 dates are invalid: {error}") from error

    expected_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    if not pd.DatetimeIndex(predictions["target_date"]).equals(expected_dates):
        raise Frozen2025ArtifactsError(
            "Frozen 2025 predictions must contain sorted unique dates for the full 2025 year."
        )
    if predictions["issue_date"].isna().any() or predictions["target_date"].isna().any():
        raise Frozen2025ArtifactsError("Frozen 2025 predictions contain missing dates.")
    if not np.array_equal(
        predictions["issue_date"].to_numpy(),
        (predictions["target_date"] - pd.Timedelta(days=1)).to_numpy(),
    ):
        raise Frozen2025ArtifactsError("Frozen 2025 issue dates must precede targets by one day.")
    if predictions["split"].isna().any() or set(predictions["split"].astype(str)) != {
        "untouched_2025_holdout"
    }:
        raise Frozen2025ArtifactsError("Frozen 2025 predictions contain an unsupported split.")
    numeric_prediction_columns = (
        "target_radiation_kwh_m2_day",
        "persistence_prediction_kwh_m2_day",
        "seasonal_mean_prediction_kwh_m2_day",
        "ridge_prediction_kwh_m2_day",
    )
    for column in numeric_prediction_columns:
        try:
            values = pd.to_numeric(predictions[column], errors="raise").to_numpy(dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise Frozen2025ArtifactsError(
                f"Frozen 2025 predictions field {column} is not numeric."
            ) from error
        if not np.isfinite(values).all():
            raise Frozen2025ArtifactsError(
                f"Frozen 2025 predictions field {column} contains non-finite values."
            )
        if (values < 0).any():
            raise Frozen2025ArtifactsError(
                f"Frozen 2025 predictions field {column} must be nonnegative."
            )
        predictions[column] = values

    _validate_frozen_forecast_metrics(forecast_metrics, predictions, expected_dates)

    decision_selection = manifest["decision_selection"]
    if not isinstance(decision_selection, dict):
        raise Frozen2025ArtifactsError("Frozen decision-selection metadata must be an object.")
    if (
        decision_selection.get("scenario_id") != "primary"
        or decision_selection.get("score_name") != "mean_shortfall_plus_mean_excess_kwh_day"
    ):
        raise Frozen2025ArtifactsError("Frozen decision-selection schema is unsupported.")
    expected_approaches = {"persistence", "seasonal_mean", "ridge"}
    decision_scenario = manifest["decision_scenario"]
    if not isinstance(decision_scenario, dict) or set(decision_scenario) != {
        "panel_area_m2",
        "efficiency",
        "derating",
        "demand_kwh",
        "capacity_kwh",
        "buffer_kwh",
    }:
        raise Frozen2025ArtifactsError("Frozen reserve scenario metadata has an invalid schema.")
    for name, value in decision_scenario.items():
        _frozen_number(value, f"Frozen reserve scenario {name}", nonnegative=True)
    if (
        _frozen_number(decision_scenario["panel_area_m2"], "Frozen panel area") <= 0
        or not 0 < _frozen_number(decision_scenario["efficiency"], "Frozen efficiency") <= 1
        or not 0 < _frozen_number(decision_scenario["derating"], "Frozen derating") <= 1
    ):
        raise Frozen2025ArtifactsError("Frozen reserve scenario physical assumptions are invalid.")
    shortfall_weight = _frozen_number(
        decision_selection.get("shortfall_weight"), "Frozen shortfall weight", nonnegative=True
    )
    excess_weight = _frozen_number(
        decision_selection.get("excess_prepared_reserve_weight"),
        "Frozen excess-reserve weight",
        nonnegative=True,
    )

    if decisions[["scenario_id", "approach"]].isna().any().any():
        raise Frozen2025ArtifactsError(
            "Frozen decision metrics contain missing scenario/approach keys."
        )
    decisions["scenario_id"] = decisions["scenario_id"].astype(str)
    decisions["approach"] = decisions["approach"].astype(str)
    expected_scenario = decision_selection["scenario_id"]
    if set(decisions["scenario_id"]) != {expected_scenario}:
        raise Frozen2025ArtifactsError(
            "Frozen decision metrics scenarios do not match the manifest."
        )
    if set(decisions["approach"]) != expected_approaches or len(decisions) != len(
        expected_approaches
    ):
        raise Frozen2025ArtifactsError(
            "Frozen decision metrics have an incomplete approach matrix."
        )
    if decisions.duplicated(["scenario_id", "approach"]).any():
        raise Frozen2025ArtifactsError(
            "Frozen decision metrics contain duplicate scenario/approach rows."
        )
    for column in FROZEN_2025_DECISION_COLUMNS - {"scenario_id", "approach", "decision_score_name"}:
        try:
            values = pd.to_numeric(decisions[column], errors="raise").to_numpy(dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise Frozen2025ArtifactsError(
                f"Frozen decision metric field {column} is not numeric."
            ) from error
        if not np.isfinite(values).all():
            raise Frozen2025ArtifactsError(
                f"Frozen decision metric field {column} contains non-finite values."
            )
        if (values < 0).any():
            raise Frozen2025ArtifactsError(
                f"Frozen decision metric field {column} must be nonnegative."
            )
        decisions[column] = values
    if decisions["decision_score_name"].astype(str).tolist() != [
        decision_selection["score_name"]
    ] * len(decisions):
        raise Frozen2025ArtifactsError("Frozen decision score names do not match the manifest.")
    if not np.allclose(decisions["shortfall_weight"], shortfall_weight, rtol=0, atol=1e-12):
        raise Frozen2025ArtifactsError("Frozen shortfall weights do not match the manifest.")
    if not np.allclose(
        decisions["excess_prepared_reserve_weight"], excess_weight, rtol=0, atol=1e-12
    ):
        raise Frozen2025ArtifactsError("Frozen excess-reserve weights do not match the manifest.")
    expected_score = (
        decisions["mean_simulated_shortfall_kwh_day"] * shortfall_weight
        + decisions["mean_excess_prepared_reserve_kwh_day"] * excess_weight
    )
    if not np.allclose(
        decisions["mean_shortfall_plus_mean_excess_kwh_day"],
        expected_score,
        rtol=1e-9,
        atol=1e-10,
    ):
        raise Frozen2025ArtifactsError(
            "Frozen decision scores do not match their component metrics."
        )
    if any(
        isinstance(value, (bool, np.bool_)) or not float(value).is_integer()
        for value in decisions["sample_count"]
    ) or set(decisions["sample_count"].astype(int)) != {len(expected_dates)}:
        raise Frozen2025ArtifactsError("Frozen decision metrics have an invalid sample count.")
    day_counts = decisions["simulated_shortfall_day_count"].to_numpy(dtype=float)
    if (
        not np.equal(day_counts, np.floor(day_counts)).all()
        or (day_counts > len(expected_dates)).any()
    ):
        raise Frozen2025ArtifactsError("Frozen decision shortfall day counts are invalid.")
    for _, row in decisions.iterrows():
        count = int(row["simulated_shortfall_day_count"])
        share = float(row["simulated_shortfall_day_share"])
        if (
            not count <= len(expected_dates)
            or not 0 <= share <= 1
            or not math.isclose(share, count / len(expected_dates), rel_tol=0, abs_tol=1e-12)
        ):
            raise Frozen2025ArtifactsError("Frozen decision shortfall counts and shares disagree.")

    return {
        "manifest": manifest,
        "predictions": predictions,
        "decisions": decisions,
        "forecast_metrics": forecast_metrics,
    }


def _show_frozen_2025(project_root: Path) -> None:
    """Show a read-only, separately labeled comparison of the untouched 2025 holdout."""
    st.title("Frozen 2025 results")
    try:
        artifacts = _load_frozen_2025(project_root)
    except Frozen2025ArtifactsError as error:
        st.error(str(error))
        return

    manifest = artifacts["manifest"]
    predictions = artifacts["predictions"]
    decisions = artifacts["decisions"]
    forecast_metrics = artifacts["forecast_metrics"]
    holdout = manifest["holdout_configuration"]
    scenario = manifest["decision_scenario"]
    st.warning(
        "Frozen historical holdout · Applied the development-frozen model and baselines to "
        "2025 without refitting or selection on these dates. This is descriptive research, "
        "not an operational forecast or measured facility performance."
    )
    st.caption(
        f"{len(predictions)} common targets · {predictions['target_date'].min():%Y-%m-%d} to "
        f"{predictions['target_date'].max():%Y-%m-%d} · source: {holdout['source']} · "
        f"reference location: {holdout['latitude']:.4f}°, {holdout['longitude']:.4f}° · "
        f"daily date basis: {holdout['time_standard']} · target: model/satellite-derived "
        "solar radiation (kWh/m²/day)."
    )

    metric_rows = []
    metrics = forecast_metrics["metrics"]
    for approach in ("ridge", "persistence", "seasonal_mean"):
        metric_rows.append(
            {
                "Approach": APPROACH_LABELS.get(approach, approach),
                "Radiation MAE (kWh/m²/day)": metrics[approach]["mae"],
                "Radiation RMSE (kWh/m²/day)": metrics[approach]["rmse"],
            }
        )
    forecast_comparison = pd.DataFrame(metric_rows)
    st.subheader("Frozen forecast accuracy and baseline comparison")
    st.dataframe(forecast_comparison, hide_index=True, width="stretch")
    st.bar_chart(forecast_comparison.set_index("Approach")["Radiation MAE (kWh/m²/day)"])

    st.subheader("Frozen monthly forecast error")
    month_keys = [f"{month:02d}" for month in range(1, 13)]
    monthly_rows = []
    for month_key in month_keys:
        row = {
            "Month": pd.Timestamp(f"2025-{month_key}-01").strftime("%B"),
            "n": metrics["ridge"]["monthly_sample_count"][month_key],
        }
        for approach, label in (
            ("ridge", "Ridge"),
            ("persistence", "Persist."),
            ("seasonal_mean", "Seasonal"),
        ):
            row[label] = metrics[approach]["monthly_mae_by_target_calendar_month"][month_key]
        monthly_rows.append(row)
    st.dataframe(pd.DataFrame(monthly_rows), hide_index=True, width="stretch")
    st.caption(
        "Cells are MAE in kWh/m²/day; n is the number of observed target days in each month."
    )

    st.subheader("Frozen seasonal forecast error")
    season_names = {
        "DJF": "DJF (Dec–Feb)",
        "MAM": "MAM (Mar–May)",
        "JJA": "JJA (Jun–Aug)",
        "SON": "SON (Sep–Nov)",
    }
    seasonal_rows = []
    for season, label in season_names.items():
        row = {
            "Season": label,
            "n": metrics["ridge"]["seasonal_sample_count"][season],
        }
        for approach, short_label in (
            ("ridge", "Ridge"),
            ("persistence", "Persist."),
            ("seasonal_mean", "Seasonal"),
        ):
            row[short_label] = metrics[approach]["seasonal_mae"][season]
        seasonal_rows.append(row)
    st.dataframe(pd.DataFrame(seasonal_rows), hide_index=True, width="stretch")
    st.caption(
        "Cells are MAE in kWh/m²/day; n is the number of observed target days in each season."
    )

    st.subheader("Frozen low-solar slice")
    low_solar_rows = []
    for approach in ("ridge", "persistence", "seasonal_mean"):
        low_solar = metrics[approach]["low_solar"]
        low_solar_rows.append(
            {
                "Approach": {
                    "ridge": "Ridge",
                    "persistence": "Persistence",
                    "seasonal_mean": "Seasonal mean",
                }[approach],
                "Days": low_solar["sample_count"],
                "MAE": low_solar["mae"],
                "RMSE": low_solar["rmse"],
                "Bias": low_solar["bias_prediction_minus_actual"],
            }
        )
    st.dataframe(pd.DataFrame(low_solar_rows), hide_index=True, width="stretch")
    low_solar_threshold = forecast_metrics["low_solar_threshold_frozen_from_training_kwh_m2_day"]
    low_solar_count = metrics["ridge"]["low_solar"]["sample_count"]
    st.caption(
        f"Training-frozen low-solar threshold: {low_solar_threshold:.4f} kWh/m²/day · "
        f"{low_solar_count} target days per approach. MAE, RMSE, and bias are in "
        "kWh/m²/day. Counts and errors are checksum-validated stored holdout metrics; "
        "this slice was not used to tune the model."
    )

    primary = decisions.loc[decisions["scenario_id"] == "primary"].copy()
    ridge_shortfall = float(
        primary.loc[primary["approach"] == "ridge", "mean_simulated_shortfall_kwh_day"].iloc[0]
    )
    persistence_shortfall = float(
        primary.loc[primary["approach"] == "persistence", "mean_simulated_shortfall_kwh_day"].iloc[
            0
        ]
    )
    primary["Approach"] = primary["approach"].map(APPROACH_LABELS)
    primary = primary.rename(
        columns={
            "mean_simulated_shortfall_kwh_day": "Mean simulated shortfall (kWh/day)",
            "mean_excess_prepared_reserve_kwh_day": "Mean excess prepared reserve (kWh/day)",
            "mean_shortfall_plus_mean_excess_kwh_day": "Illustrative equal-weight score (kWh/day)",
        }
    )
    st.subheader("Simulated reserve trade-off · primary frozen scenario")
    st.caption(
        "Frozen scenario settings · "
        f"demand {scenario['demand_kwh']:g} kWh/day · reserve capacity "
        f"{scenario['capacity_kwh']:g} kWh · panel area {scenario['panel_area_m2']:g} m² · "
        f"conversion efficiency {scenario['efficiency']:g} · derating "
        f"{scenario['derating']:g} · buffer {scenario['buffer_kwh']:g} kWh/day. "
        "These stored settings are separate from the live editable assumptions on the "
        "next-day prediction page."
    )
    st.dataframe(
        primary[
            [
                "Approach",
                "Mean simulated shortfall (kWh/day)",
                "Mean excess prepared reserve (kWh/day)",
                "Illustrative equal-weight score (kWh/day)",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Predicted solar contribution informs reserve preparation. The displayed simulated "
        "shortfall and excess are computed afterward against the observed 2025 radiation "
        "target under these hypothetical settings; they are not measured facility outcomes."
    )
    st.markdown(
        f"Ridge radiation MAE is **{metrics['ridge']['mae']:.3f}**, compared with "
        f"**{metrics['persistence']['mae']:.3f}** for persistence and "
        f"**{metrics['seasonal_mean']['mae']:.3f}** for the training-only seasonal mean. "
        f"Ridge's simulated shortfall is **{ridge_shortfall:.3f} kWh/day**, higher than "
        f"persistence at **{persistence_shortfall:.3f} kWh/day**. The equal-weight "
        "shortfall-plus-excess score is hypothetical; it is not a measured cost or "
        "reliability target."
    )
    chart_columns = {
        "target_radiation_kwh_m2_day": "Observed NASA POWER target",
        "ridge_prediction_kwh_m2_day": "Ridge prediction",
        "persistence_prediction_kwh_m2_day": "Persistence prediction",
        "seasonal_mean_prediction_kwh_m2_day": "Seasonal-mean prediction",
    }
    st.line_chart(
        predictions.set_index("target_date")[
            [
                "target_radiation_kwh_m2_day",
                "ridge_prediction_kwh_m2_day",
                "persistence_prediction_kwh_m2_day",
                "seasonal_mean_prediction_kwh_m2_day",
            ]
        ].rename(columns=chart_columns),
        y_label="Solar radiation (kWh/m²/day)",
    )
    st.subheader("Unfavorable frozen forecast examples")
    frozen_forecasts = predictions.rename(columns={"target_radiation_kwh_m2_day": RADIATION_COLUMN})
    example_rows = []
    for approach in ("ridge", "persistence", "seasonal_mean"):
        examples = failure_examples(frozen_forecasts, approach)
        if examples.empty:
            continue
        examples = examples.rename(
            columns={
                "example_type": "Example",
                "target_date": "Target date",
                RADIATION_COLUMN: "Observed target (kWh/m²/day)",
                PREDICTION_COLUMNS[approach]: "Predicted radiation (kWh/m²/day)",
                "prediction_error_kwh_m2_day": "Prediction minus observed (kWh/m²/day)",
            }
        )
        examples.insert(0, "Approach", APPROACH_LABELS[approach])
        example_rows.append(examples)
    if example_rows:
        st.caption(
            "Largest overprediction and underprediction for each approach, selected from the "
            "checksum-verified stored 2025 predictions after the holdout was scored. These "
            "examples describe misses; they were not used for model selection or tuning."
        )
        st.dataframe(pd.concat(example_rows, ignore_index=True), hide_index=True, width="stretch")
    else:
        st.info("No nonzero forecast-error examples are present in the verified holdout rows.")
    st.caption(
        f"Frozen run {manifest.get('run_id', 'frozen-test-2025')} · source artifact files "
        "were checksum-verified before display."
    )


def _format_kwh_day(value: float) -> str:
    """Keep simulated-energy metric values compact enough for a narrow Streamlit card."""
    return f"{value:.2f} kWh/day"


@st.cache_data(show_spinner=False)
def _cached_artifacts(
    root_text: str,
    code_root_text: str,
    signature: tuple[tuple[str, str], ...],
    experiment_id: str | None = None,
) -> DemoArtifacts:
    """Cache local reads by content signature; never fetch, fit, or rewrite artifacts."""
    # Keep this as a normal argument so Streamlit includes it in the cache key.
    del signature
    return load_demo_artifacts(
        Path(root_text), code_root=Path(code_root_text), experiment_id=experiment_id
    )


def _show_recovery_guidance() -> None:
    st.markdown("**Restore the trusted shipped validation and model**")
    st.caption("Run from the project root. The app never restores or rewrites artifacts itself.")
    st.code(
        "git restore --source=HEAD -- artifacts/validation models/ridge_model.json",
        language="bash",
    )
    st.markdown("**Create a separate validation experiment, then select it above**")
    st.code(
        "python -m sunreserve validate --run-id recovery-2026-11-19",
        language="bash",
    )
    st.caption(
        "The required run ID creates immutable output under "
        "`artifacts/experiments/recovery-2026-11-19/`; it does not replace the shipped default "
        "artifacts. Choose the new experiment in the Validation artifact set control after the "
        "command finishes. Use a different safe ID if that one already exists."
    )


def _select_validation_experiment(project_root: Path) -> str | None:
    """Let both editable modes use the shipped model or one verified isolated run."""
    experiment_ids = list_validation_experiments(project_root)
    options = ("shipped", *experiment_ids)
    selected_source = st.sidebar.selectbox(
        "Validation artifact set",
        options=options,
        index=0,
        format_func=lambda value: (
            "Shipped defaults" if value == "shipped" else f"Experiment: {value}"
        ),
        key="sunreserve_validation_artifact_set",
        help=(
            "Choose the shipped validation and model, or a verified isolated validation "
            "experiment. Frozen 2025 results always use their separate frozen artifacts."
        ),
    )
    return selected_source if selected_source in options[1:] else None


def _show_load_failure(error: ArtifactError) -> None:
    st.title("SunReserve historical replay")
    st.warning(f"Cached validation data is {error.category}.")
    st.write(str(error))
    _show_recovery_guidance()
    st.caption("Launching this page never runs those commands or makes a network request.")


def _scenario_labels(metrics: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for scenario_id, group in metrics.groupby("scenario_id", sort=False):
        row = group.iloc[0]
        if str(scenario_id) == "primary":
            label = "Primary hypothetical scenario"
        else:
            label = f"Sensitivity: {scenario_id}"
        label += (
            f" · demand {row['demand_kwh']:g} kWh · capacity {row['capacity_kwh']:g} kWh"
            f" · buffer {row['buffer_kwh']:g} kWh"
        )
        labels[str(scenario_id)] = label
    return labels


def _show_replay(artifacts: DemoArtifacts) -> None:
    manifest = artifacts.manifest
    quality = artifacts.quality
    selected_model = manifest["model_selection"]["selected_model"]
    forecast_metrics = manifest["model_selection"]["validation_metrics"]

    st.title("SunReserve historical replay")
    st.warning(
        "2022 validation replay · Historical research only. This is not a validated "
        "operational controller or a real school deployment."
    )
    st.caption(
        f"Source: {manifest['configuration']['source']} · reference coordinates: "
        f"{manifest['configuration']['latitude']:.4f}°, "
        f"{manifest['configuration']['longitude']:.4f}° · daily date basis: "
        f"{manifest['configuration']['time_standard']}. NASA POWER radiation is "
        "model/satellite-derived in kWh/m²/day; electricity, prepared reserve, shortfall, "
        "and excess are hypothetical simulated kWh/day quantities."
    )
    st.markdown(
        """
        <style>
        @media (max-width: 640px) {
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0.5rem !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                max-width: 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    scenario_labels = _scenario_labels(artifacts.decisions)
    scenario_ids = list(scenario_labels)
    with st.sidebar:
        st.header("Explore validation rows")
        model = st.selectbox(
            "Forecast approach",
            options=list(APPROACHES),
            index=list(APPROACHES).index(selected_model),
            format_func=lambda value: APPROACH_LABELS[value],
        )
        scenario_id = st.selectbox(
            "Saved reserve scenario",
            options=scenario_ids,
            index=scenario_ids.index("primary") if "primary" in scenario_ids else 0,
            format_func=lambda value: scenario_labels[value],
        )
        target_date = st.date_input(
            "Validation target date",
            value=artifacts.dates[0],
            min_value=artifacts.dates[0],
            max_value=artifacts.dates[-1],
            format="YYYY-MM-DD",
        )

    if target_date not in artifacts.dates:
        st.error("The selected date is not present in the cached validation predictions.")
        return

    prediction_column = PREDICTION_COLUMNS[model] if model in PREDICTION_COLUMNS else None
    selected_forecast = artifacts.forecasts.loc[
        artifacts.forecasts["target_date"].dt.date == target_date
    ].iloc[0]
    selected_decision = artifacts.decisions.loc[
        (artifacts.decisions["scenario_id"].astype(str) == scenario_id)
        & (artifacts.decisions["approach"] == model)
        & (artifacts.decisions["target_date"].dt.date == target_date)
    ]
    if selected_decision.empty:
        st.error("No saved reserve decision exists for this date, approach, and scenario.")
        return
    decision = selected_decision.iloc[0]
    forecast_value = (
        float(selected_forecast[prediction_column]) if prediction_column is not None else None
    )

    st.subheader(f"Target date · {target_date:%Y-%m-%d}")
    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Observed NASA POWER radiation (target)",
        f"{selected_forecast[RADIATION_COLUMN]:.2f} kWh/m²/day",
    )
    metric_columns[1].metric(
        "Predicted solar radiation",
        "Not forecast-based" if forecast_value is None else f"{forecast_value:.2f} kWh/m²/day",
    )
    reserve_columns = st.columns(2)
    reserve_columns[0].metric(
        "Prepared reserve (forecast, after capacity cap)",
        _format_kwh_day(decision["prepared_reserve_kwh"]),
    )
    reserve_columns[1].metric(
        "Simulated realized electricity proxy",
        _format_kwh_day(decision["actual_generation_proxy_kwh_day"]),
        help=(
            "Observed target radiation × assumed panel area × efficiency × derating. "
            "This is a post-hoc simulated proxy, not measured generation."
        ),
    )
    energy_columns = st.columns(2)
    energy_columns[0].metric(
        "Simulated predicted electricity proxy",
        (
            "Not forecast-based"
            if model == "always_prepare_backup"
            else _format_kwh_day(decision["predicted_generation_proxy_kwh_day"])
        ),
        help="Predicted radiation × assumed panel area × efficiency × derating.",
    )
    energy_columns[1].metric(
        "Post-hoc simulated shortfall",
        _format_kwh_day(decision["simulated_shortfall_kwh"]),
        help=(
            "Uses observed target radiation after the date occurred, plus the forecast-based "
            "reserve prepared under the saved capacity limit."
        ),
    )
    excess_columns = st.columns(2)
    excess_columns[0].metric(
        "Excess prepared reserve",
        _format_kwh_day(decision["excess_prepared_reserve_kwh"]),
        help="Prepared backup beyond realized residual demand; not surplus solar generation.",
    )
    st.caption(
        f"Assumed demand {decision['demand_kwh']:g} kWh/day · reserve capacity "
        f"{decision['capacity_kwh']:g} kWh/day · buffer {decision['buffer_kwh']:g} kWh/day. "
        f"Capacity-limited shortfall floor: {decision['capacity_limited_shortfall_kwh']:.2f} "
        "kWh/day. All assumptions are hypothetical."
    )
    st.caption(
        "Predicted radiation guides reserve preparation; simulated realized electricity, "
        "shortfall, and excess are post-hoc outcomes calculated from the observed radiation "
        "target in validation. "
        "Prepared reserve is the forecast-based amount after the capacity cap. These outcomes "
        "are not available at issue time and are not measured facility performance."
    )

    st.subheader("2022 radiation forecasts")
    chart_column = PREDICTION_COLUMNS[model] if prediction_column else None
    if chart_column:
        chart = artifacts.forecasts.set_index("target_date")[[RADIATION_COLUMN, chart_column]]
        chart = chart.rename(
            columns={
                RADIATION_COLUMN: "NASA POWER radiation",
                chart_column: f"{APPROACH_LABELS[model]} prediction",
            }
        )
        st.line_chart(chart, y_label="Solar radiation (kWh/m²/day)")
    else:
        st.info("The always-prepare-backup reference has no radiation forecast to chart.")

    st.subheader("Baseline and reserve comparison")
    scenario_metrics = artifacts.decision_metrics.loc[
        artifacts.decision_metrics["scenario_id"].astype(str) == scenario_id
    ].copy()
    scenario_metrics["Approach"] = scenario_metrics["approach"].map(APPROACH_LABELS)
    scenario_metrics["Validation forecast MAE (kWh/m²/day)"] = scenario_metrics["approach"].map(
        lambda name: forecast_metrics.get(name, {}).get("mae")
    )
    comparison = scenario_metrics[
        [
            "Approach",
            "Validation forecast MAE (kWh/m²/day)",
            "mean_simulated_shortfall_kwh_day",
            "simulated_shortfall_day_share",
            "mean_excess_prepared_reserve_kwh_day",
            "total_capacity_limited_shortfall_kwh",
        ]
    ].rename(
        columns={
            "mean_simulated_shortfall_kwh_day": "Mean simulated shortfall (kWh/day)",
            "simulated_shortfall_day_share": "Shortfall day share",
            "mean_excess_prepared_reserve_kwh_day": "Mean excess prepared reserve (kWh/day)",
            "total_capacity_limited_shortfall_kwh": "Capacity-limited shortfall total (kWh)",
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    st.subheader("Largest forecast errors")
    if prediction_column:
        examples = failure_examples(artifacts.forecasts, model)
        examples = examples.rename(
            columns={
                RADIATION_COLUMN: "NASA POWER radiation (kWh/m²/day)",
                prediction_column: "Predicted radiation (kWh/m²/day)",
                "prediction_error_kwh_m2_day": "Prediction minus observed (kWh/m²/day)",
            }
        )
        st.caption(
            "Deterministic examples from the displayed model: the largest positive and "
            "largest negative prediction errors across the 2022 validation dates."
        )
        if examples.empty:
            st.info("This model has no strictly positive or negative validation error example.")
        else:
            st.dataframe(examples, hide_index=True, width="stretch")
    else:
        st.info("The always-prepare-backup reference does not produce forecast-error examples.")

    with st.expander("Source, missingness, and assumptions"):
        location = quality.get("requested_location", {})
        st.write(
            f"Source: {quality.get('source', 'Unknown')} · NASA POWER daily "
            "model/satellite-derived data at "
            f"{location.get('latitude')}° latitude, {location.get('longitude')}° longitude · "
            f"time standard: {quality.get('time_standard', 'Unknown')}."
        )
        st.write(
            f"Source coverage: {quality.get('requested_start')} through "
            f"{quality.get('requested_end')} · "
            f"{quality.get('calendar_days')} calendar days · {quality.get('complete_case_days')} "
            f"complete-case days · {quality.get('excluded_days')} excluded days."
        )
        st.write("Units in source quality report:")
        st.json(quality.get("units", {}), expanded=False)
        st.write("Missing values by variable:")
        st.json(quality.get("missing_counts_by_variable", {}), expanded=False)
        st.markdown(
            "This replay uses historical daily radiation and does not verify real-time data "
            "availability. The electricity proxy and reserve assumptions do not demonstrate "
            "rooftop output, measured school demand, battery behavior, deployment, savings, "
            "or emissions reductions. See `docs/assumptions.md` for formulas and limits."
        )
        st.caption(
            f"Validation artifacts created: {manifest.get('created_utc')} · "
            f"code revision: {manifest.get('code_revision', 'unknown')} · "
            "selected by validation MAE: "
            f"{selected_model}."
        )


def _sample_observation_window(paths: ProjectPaths) -> pd.DataFrame:
    """Prepopulate the editor with the latest local source rows, if present."""
    columns = ["date", RADIATION, "T2M", "RH2M"]
    try:
        daily = pd.read_csv(paths.daily_path, parse_dates=["date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame({column: [None] * 7 for column in columns})
    if not set(columns) <= set(daily.columns):
        return pd.DataFrame({column: [None] * 7 for column in columns})
    window = daily.loc[:, columns].tail(7).reset_index(drop=True)
    if len(window) != 7:
        return pd.DataFrame({column: [None] * 7 for column in columns})
    window["date"] = pd.to_datetime(window["date"]).dt.date
    return window


def _observation_csv_template() -> str:
    """Return a blank seven-row template without presenting fabricated data as history."""
    template = pd.DataFrame({column: [""] * 7 for column in OBSERVATION_COLUMNS})
    return template.to_csv(index=False, lineterminator="\n")


def _set_observation_editor_data(frame: pd.DataFrame) -> None:
    """Set programmatic editor input and remount its widget without assigning its widget key."""
    st.session_state[OBSERVATION_EDITOR_DATA_KEY] = frame.copy()
    st.session_state[OBSERVATION_EDITOR_VERSION_KEY] = (
        int(st.session_state.get(OBSERVATION_EDITOR_VERSION_KEY, 0)) + 1
    )


def _observation_origin(frame: pd.DataFrame) -> str:
    """Distinguish a complete local history window from an empty template."""
    if len(frame) == 7 and set(OBSERVATION_COLUMNS) <= set(frame.columns):
        if not frame[OBSERVATION_COLUMNS].isna().any(axis=None):
            return "local_historical_example"
    return "blank_template"


def _observation_origin_caption(origin: str) -> str:
    descriptions = {
        "local_historical_example": (
            "Input origin: local historical example from prepared daily data; it is not a live "
            "feed and was not checked for availability at the original forecast issue time."
        ),
        "csv_import": (
            "Input origin: user-imported CSV; its original data source and measurement time "
            "are not independently verified here."
        ),
        "user_edited": (
            "Input origin: user-edited observations; their source and measurement time are "
            "not independently verified here."
        ),
        "guided_example": (
            "Input origin: guided synthetic change for demonstration; it is not an observation."
        ),
        "blank_template": "Input origin: blank template; enter seven compatible daily rows.",
    }
    description = descriptions.get(origin, descriptions["user_edited"])
    return (
        f"{description} Model reference location: Bengaluru, India "
        "(12.9716° N, 77.5946° E); date labels are treated as UTC daily calendar dates. "
        "Units: solar radiation kWh/m²/day, temperature °C, and relative humidity %."
    )


def _parse_observation_csv(payload: bytes) -> pd.DataFrame:
    """Parse a strict seven-row CSV and report input errors at their source row."""
    try:
        uploaded = pd.read_csv(io.BytesIO(payload), dtype={"date": "string"})
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"CSV could not be read: {error}") from error
    if list(uploaded.columns) != OBSERVATION_COLUMNS:
        expected = ", ".join(OBSERVATION_COLUMNS)
        received = ", ".join(str(column) for column in uploaded.columns)
        raise ValueError(f"CSV headers must be exactly: {expected}. Found: {received or 'none'}.")
    if len(uploaded) != 7:
        raise ValueError(f"CSV must contain exactly 7 observation rows; found {len(uploaded)}.")

    frame = observation_frame_from_records(
        uploaded.to_dict(orient="records"), row_label="CSV row"
    ).reset_index(names="date")
    frame["date"] = frame["date"].dt.date
    return frame.loc[:, OBSERVATION_COLUMNS]


def _edited_window_frame(edited: pd.DataFrame) -> pd.DataFrame:
    """Convert Streamlit editor rows to the strict inference input contract."""
    records = edited.to_dict(orient="records")
    for position, record in enumerate(records, start=1):
        raw_date = record.get("date")
        if raw_date is None or pd.isna(raw_date):
            raise ValueError(f"observation {position} is missing its date")
        try:
            stamp = pd.Timestamp(raw_date)
        except (TypeError, ValueError) as error:
            raise ValueError(f"observation {position} has an invalid date") from error
        if stamp.tzinfo is not None or stamp != stamp.normalize():
            raise ValueError(
                f"observation {position} date must be a timezone-naive calendar date at midnight"
            )
        record["date"] = stamp.date().isoformat()
    return observation_frame_from_records(records)


def _snapshot_value(value: object) -> object:
    """Convert editor values into stable JSON-safe values without hiding invalid edits."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        marker = (
            "nan"
            if math.isnan(value)
            else "positive_infinity"
            if value > 0
            else "negative_infinity"
        )
        return {"$nonfinite_float": marker}
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _input_snapshot(edited: pd.DataFrame) -> list[dict[str, object]]:
    """Capture the exact current observation rows, including incomplete editor values."""
    return [
        {str(key): _snapshot_value(value) for key, value in row.items()}
        for row in edited.to_dict(orient="records")
    ]


def _prediction_fingerprint(
    observations: list[dict[str, object]],
    assumptions: dict[str, object],
    model_identity: dict[str, object],
    source_kind: str,
) -> str:
    payload = {
        "observations": observations,
        "assumptions": assumptions,
        "model": model_identity,
        "source_kind": source_kind,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Store tabular outputs as JSON-safe dictionaries in Streamlit session state."""
    return _input_snapshot(frame)


def _prediction_decision_table(saved: dict[str, object]) -> pd.DataFrame:
    """Describe forecast-only reserve decisions without implying observed outcomes."""
    plans = pd.DataFrame(saved["reserve_plans"]).copy()
    demand = plans["demand_kwh"].astype(float)
    predicted_solar = plans["predicted_generation_proxy_kwh_day"].astype(float)
    prepared = plans["prepared_reserve_kwh"].astype(float)
    buffer = plans["buffer_kwh"].astype(float)
    plans["backup_need_before_capacity_cap_kwh"] = (demand - predicted_solar + buffer).clip(
        lower=0.0
    )
    plans["predicted_remaining_gap_kwh"] = (demand - predicted_solar - prepared).clip(lower=0.0)
    return plans


def _prediction_export_payload(
    saved: dict[str, object], *, result_status: str
) -> dict[str, object]:
    """Build the shared portable result for the saved inference and current UI freshness."""
    plans = _prediction_decision_table(saved)
    prediction = saved["prediction"]
    assumptions = saved["assumptions"]
    return build_prediction_export(
        generated_at_utc=str(saved["generated_at_utc"]),
        issue_date=str(prediction["issue_date"]),
        target_date=str(prediction["target_date"]),
        source_kind=str(saved.get("observation_origin", "user_edited")),
        observations=saved["observations"],
        assumptions={
            "daily_demand_kwh": assumptions["demand"],
            "reserve_capacity_kwh": assumptions["capacity"],
            "reserve_buffer_kwh": assumptions["buffer"],
            "panel_area_m2": assumptions["area"],
            "conversion_efficiency": assumptions["efficiency"],
            "derating_factor": assumptions["derating"],
        },
        model_identity=saved["model_identity"],
        package_version=str(saved["package_version"]),
        package_source=saved["package_source_identity"],
        entrypoint_identity=saved["entrypoint_identity"],
        forecasts={
            "ridge_radiation_kwh_m2_day": prediction["radiation_kwh_m2_day"],
            "persistence_radiation_kwh_m2_day": prediction["persistence_kwh_m2_day"],
            "seasonal_mean_radiation_kwh_m2_day": prediction["seasonal_mean_kwh_m2_day"],
        },
        reserve_outputs=_json_records(plans),
        features=prediction["features"],
        warnings=prediction["warnings"],
        input_validation_policy=OBSERVATION_POLICY_NOTE,
        result_status=result_status,
    )


def _show_saved_prediction(saved: dict[str, object], *, result_status: str) -> None:
    prediction = saved["prediction"]
    plans = _prediction_decision_table(saved)
    forecasts = [
        prediction["radiation_kwh_m2_day"],
        prediction["persistence_kwh_m2_day"],
        prediction["seasonal_mean_kwh_m2_day"],
    ]
    st.subheader(f"Fresh prediction for {prediction['target_date']}")
    for warning in prediction.get("warnings", []):
        st.warning(warning)
    st.metric("Ridge predicted solar radiation", f"{forecasts[0]:.2f} kWh/m²/day")
    ridge_plan = plans.iloc[0]
    st.subheader("Predicted reserve summary · Ridge")
    st.caption(_observation_origin_caption(str(saved.get("observation_origin", "user_edited"))))
    st.caption(
        "Predicted solar contribution is a radiation-based electricity proxy. Backup need "
        "before the capacity cap accounts for assumed demand, predicted solar contribution, "
        "and buffer; planned reserve is limited by the assumed capacity. The remaining "
        "predicted gap is demand left uncovered after predicted solar and prepared reserve."
    )
    summary = pd.DataFrame(
        [
            {
                "Quantity": "Assumed daily demand",
                "Value": ridge_plan["demand_kwh"],
                "Units": "kWh/day",
            },
            {
                "Quantity": "Predicted solar electricity contribution (proxy)",
                "Value": ridge_plan["predicted_generation_proxy_kwh_day"],
                "Units": "kWh/day",
            },
            {
                "Quantity": "Predicted backup need before capacity cap",
                "Value": ridge_plan["backup_need_before_capacity_cap_kwh"],
                "Units": "kWh/day",
            },
            {
                "Quantity": "Predicted planned reserve after capacity cap",
                "Value": ridge_plan["prepared_reserve_kwh"],
                "Units": "kWh/day",
            },
            {
                "Quantity": "Predicted remaining gap after solar and reserve",
                "Value": ridge_plan["predicted_remaining_gap_kwh"],
                "Units": "kWh/day",
            },
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    capacity = float(ridge_plan["capacity_kwh"])
    required = float(ridge_plan["backup_need_before_capacity_cap_kwh"])
    if capacity == 0:
        st.warning(
            "Reserve capacity is zero, so no backup can be prepared. The predicted remaining "
            f"gap is {ridge_plan['predicted_remaining_gap_kwh']:.2f} kWh/day "
            "under these assumptions."
        )
    elif required > capacity:
        st.warning(
            f"Predicted backup need ({required:.2f} kWh/day) exceeds available capacity "
            f"({capacity:.2f} kWh). The planned reserve is capped at capacity; the predicted "
            f"remaining gap is {ridge_plan['predicted_remaining_gap_kwh']:.2f} kWh/day."
        )
    else:
        st.success("Available capacity covers the predicted backup need under these assumptions.")
    st.info(
        "Forecast uncertainty remains: actual solar contribution can differ from the prediction, "
        "so a shortage can still occur even when capacity covers the predicted need."
    )
    identity = saved["model_identity"]
    st.caption(
        f"Model {identity['model_id']}; trained on {identity['training_rows']} training rows "
        f"through {identity['training_end']}. Model SHA-256: {identity['artifact_sha256']}."
    )
    if saved.get("decision_selection", {}).get("selected_model"):
        st.caption(
            "Development-only reserve objective selected "
            f"{saved['decision_selection']['selected_model']} on the 2022 validation period. "
            "This is a research ranking, not an operational reliability guarantee."
        )
    st.subheader("Fresh inference and baseline reserve comparison")
    st.dataframe(
        pd.DataFrame(
            {
                "Approach": [
                    "Trained Ridge model",
                    "Persistence baseline",
                    "Seasonal-mean baseline",
                ],
                "Predicted radiation (kWh/m²/day)": forecasts,
                "Predicted electricity proxy (kWh/day)": plans[
                    "predicted_generation_proxy_kwh_day"
                ].to_list(),
                "Backup need before capacity cap (kWh/day)": plans[
                    "backup_need_before_capacity_cap_kwh"
                ].to_list(),
                "Planned reserve after capacity cap (kWh/day)": plans[
                    "prepared_reserve_kwh"
                ].to_list(),
                "Predicted remaining gap (kWh/day)": plans["predicted_remaining_gap_kwh"].to_list(),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "The target day's actual radiation is not yet supplied, so realized shortfall and "
        "excess prepared reserve are not calculated in prediction mode. No forecast-data "
        "availability or facility performance is established."
    )
    with st.expander("Feature and model details"):
        st.write(
            f"Issue date: {prediction['issue_date']} · target date: {prediction['target_date']} · "
            f"model: {prediction['model_id']} · training rows: {identity['training_rows']}."
        )
        st.json(prediction["features"], expanded=False)
        st.write(
            "Exact input observations and hypothetical reserve assumptions are saved "
            "with this result."
        )
        st.markdown(
            "Training-only median imputation, standard scaling, and Ridge coefficients are "
            "loaded from the versioned numeric model bundle. Input window values are not "
            "reused cached predictions."
        )
    assumptions = saved["assumptions"]
    st.caption(
        f"Assumptions: demand {assumptions['demand']:g} kWh/day, reserve capacity "
        f"{assumptions['capacity']:g} kWh, buffer {assumptions['buffer']:g} kWh, "
        f"area {assumptions['area']:g} m², efficiency {assumptions['efficiency']:g}, "
        f"derating {assumptions['derating']:g}. All are hypothetical, not measured facility values."
    )
    if (
        not saved.get("generated_at_utc")
        or not saved.get("entrypoint_identity")
        or not saved.get("package_version")
        or not saved.get("package_source_identity")
    ):
        st.info(
            "This saved result predates the JSON export provenance contract. Run a fresh "
            "prediction to create a complete export."
        )
    else:
        export_data = json.dumps(
            _prediction_export_payload(saved, result_status=result_status),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        st.download_button(
            "Download complete prediction result (JSON)",
            data=export_data,
            file_name=f"sunreserve-prediction-{prediction['target_date']}.json",
            mime="application/json",
            help=(
                "Includes exact observation rows, assumptions, model identity, forecasts, "
                "and reserve outputs."
            ),
        )


def _show_prediction(project_root: Path, experiment_id: str | None = None) -> None:
    """Collect seven observed rows and run new inference from the saved fitted model."""
    paths = ProjectPaths(project_root)
    st.title("SunReserve next-day prediction")
    st.warning(
        "Research prototype · Historical observation dates are not verified as available "
        "at forecast issue time. No live weather feed or operational controller is provided."
    )
    st.caption(
        "Enter seven consecutive daily observations ending on the issue date. The trained "
        "Ridge model predicts next-calendar-day NASA POWER solar radiation; converted "
        "electricity and prepared reserve are hypothetical simulated quantities."
    )
    st.caption(OBSERVATION_POLICY_NOTE)
    config = ExperimentConfig()
    try:
        if experiment_id is None:
            model, decision_selection = load_project_model(project_root)
        else:
            model, decision_selection = load_project_model(
                project_root,
                experiment_id=experiment_id,
                expected_context=config,
            )
    except ModelArtifactError as error:
        st.error(f"Trained model unavailable: {error}")
        _show_recovery_guidance()
        st.caption("Prediction mode requires a model from the selected validation artifacts.")
        return

    with st.sidebar:
        st.header("Hypothetical reserve assumptions")
        demand = st.number_input(
            "Assumed daily demand (kWh)", min_value=0.01, value=config.daily_demand_kwh
        )
        capacity = st.number_input(
            "Assumed reserve capacity (kWh)", min_value=0.0, value=config.reserve_cap_kwh
        )
        buffer = st.number_input(
            "Assumed reserve buffer (kWh)", min_value=0.0, value=config.reserve_buffer_kwh
        )
        area = st.number_input(
            "Illustrative panel area (m²)", min_value=0.01, value=config.panel_area_m2
        )
        efficiency = st.number_input(
            "Illustrative conversion efficiency",
            min_value=0.01,
            max_value=1.0,
            value=config.conversion_efficiency,
        )
        derating = st.number_input(
            "Illustrative derating factor",
            min_value=0.01,
            max_value=1.0,
            value=config.derating_factor,
        )

    st.subheader("Recent daily observation window")
    editor = _sample_observation_window(paths)
    if OBSERVATION_EDITOR_DATA_KEY not in st.session_state:
        st.session_state[OBSERVATION_EDITOR_DATA_KEY] = editor.copy()
        st.session_state[OBSERVATION_ORIGIN_KEY] = _observation_origin(editor)
    if OBSERVATION_EDITOR_VERSION_KEY not in st.session_state:
        st.session_state[OBSERVATION_EDITOR_VERSION_KEY] = 0
    if OBSERVATION_ORIGIN_KEY not in st.session_state:
        st.session_state[OBSERVATION_ORIGIN_KEY] = "user_edited"
    default_dates = editor["date"].dropna()
    date_span = (
        f"{default_dates.iloc[0]} to {default_dates.iloc[-1]}"
        if len(default_dates)
        else "not available"
    )
    with st.expander("Input guide · location, units, and compatible observations"):
        st.markdown(
            "**Reference location:** Bengaluru, India (12.9716° N, 77.5946° E). "
            "Enter seven consecutive daily rows ending on the issue date."
        )
        st.markdown(
            f"**Historical example dates:** {date_span}. The prefilled rows are a local "
            "historical example, not a live feed and not proof those values were available "
            "on the issue date."
        )
        st.markdown(
            "**Columns and units:** `date` is YYYY-MM-DD; `ALLSKY_SFC_SW_DWN` is daily "
            "NASA POWER solar radiation in kWh/m²/day; `T2M` is 2 m air temperature in °C; "
            "`RH2M` is relative humidity in percent. Dates must be consecutive."
        )
        st.markdown(
            f"**Model period:** {model.training_start} through {model.training_end} "
            f"({model.training_rows:,} training rows). The model predicts next-day radiation; "
            "solar-to-electricity conversion and reserve settings are hypothetical."
        )
        st.markdown(
            "To use compatible observations, select the daily NASA POWER parameters above "
            "for the Bengaluru coordinates and seven consecutive dates, then confirm that "
            "each value was actually available by the forecast issue time. NASA POWER data "
            "can be delayed, so historical values do not establish live availability. "
            "See the [NASA POWER Daily API documentation]"
            "(https://power.larc.nasa.gov/docs/services/api/temporal/daily/)."
        )

    st.download_button(
        "Download blank seven-row CSV template",
        data=_observation_csv_template(),
        file_name="sunreserve-observations-template.csv",
        mime="text/csv",
        help="Fill all seven consecutive rows using the displayed column names and units.",
    )
    upload_version = int(st.session_state.get(CSV_UPLOAD_VERSION_KEY, 0))
    uploaded = st.file_uploader(
        "Import seven compatible observations from CSV",
        type=["csv"],
        key=f"{CSV_UPLOAD_KEY}_{upload_version}",
        help=f"Required columns: {', '.join(OBSERVATION_COLUMNS)}. Dates use YYYY-MM-DD.",
    )
    uploaded_digest = None
    if uploaded is not None:
        upload_payload = uploaded.getvalue()
        uploaded_digest = hashlib.sha256(upload_payload).hexdigest()
        if st.session_state.get(CSV_UPLOAD_PROCESSED_KEY) != uploaded_digest:
            try:
                imported = _parse_observation_csv(upload_payload)
            except ValueError as error:
                st.session_state[CSV_UPLOAD_STATUS_KEY] = (uploaded_digest, str(error), False)
            else:
                _set_observation_editor_data(imported)
                st.session_state[OBSERVATION_ORIGIN_KEY] = "csv_import"
                st.session_state[CSV_UPLOAD_STATUS_KEY] = (
                    uploaded_digest,
                    "Imported seven observations. Review the rows, then run prediction when ready.",
                    True,
                )
            st.session_state[CSV_UPLOAD_PROCESSED_KEY] = uploaded_digest
        status = st.session_state.get(CSV_UPLOAD_STATUS_KEY)
        if status and status[0] == uploaded_digest:
            (st.success if status[2] else st.error)(status[1])

    def reset_to_historical_example() -> None:
        _set_observation_editor_data(editor)
        st.session_state[OBSERVATION_ORIGIN_KEY] = _observation_origin(editor)
        st.session_state[CSV_UPLOAD_VERSION_KEY] = (
            int(st.session_state.get(CSV_UPLOAD_VERSION_KEY, 0)) + 1
        )
        st.session_state[CSV_UPLOAD_PROCESSED_KEY] = None
        st.session_state[CSV_UPLOAD_STATUS_KEY] = None
        st.session_state[GUIDED_EXAMPLE_KEY] = False

    reset_column, guide_column = st.columns(2)
    reset_column.button(
        "Reset to historical example",
        key="sunreserve_reset_example",
        on_click=reset_to_historical_example,
    )
    with guide_column:
        st.write("Guided example")
        st.caption(
            "Change only the last day's radiation by +1.0 kWh/m²/day to see how the "
            "forecast responds. This is an input demonstration, not an observation."
        )
        if st.button("Apply the guided input change", key="sunreserve_guided_change"):
            guided = pd.DataFrame(st.session_state[OBSERVATION_EDITOR_DATA_KEY]).copy()
            try:
                current = float(guided.iloc[-1][RADIATION])
            except (IndexError, TypeError, ValueError) as error:
                st.error(
                    "Cannot apply the guided change: final-row radiation must be numeric "
                    f"({error})."
                )
            else:
                if not math.isfinite(current) or current < 0:
                    st.error(
                        "Cannot apply the guided change: final-row radiation must be finite "
                        "and nonnegative."
                    )
                else:
                    guided.loc[guided.index[-1], RADIATION] = current + 1.0
                    _set_observation_editor_data(guided)
                    st.session_state[OBSERVATION_ORIGIN_KEY] = "guided_example"
                    st.session_state[GUIDED_EXAMPLE_KEY] = True

    with st.expander("How to read the result and where to compare it"):
        st.markdown(
            "The result compares the trained Ridge model with persistence and the "
            "training-only seasonal baseline. [Jump to the comparison below]"
            "(#fresh-inference-and-baseline-reserve-comparison)."
        )
        st.markdown(
            "To review the frozen historical evaluation, select **Frozen 2025 results** "
            "in the sidebar or read the [2025 results and baseline comparison]"
            "(https://github.com/icecold009/sunreserve/blob/main/README.md#frozen-2025-results)."
        )

    editor_version = int(st.session_state[OBSERVATION_EDITOR_VERSION_KEY])
    editor_before_change = pd.DataFrame(st.session_state[OBSERVATION_EDITOR_DATA_KEY]).copy()
    edited = st.data_editor(
        st.session_state[OBSERVATION_EDITOR_DATA_KEY],
        key=f"{OBSERVATION_EDITOR_KEY}_{editor_version}",
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        column_config={
            "date": st.column_config.DateColumn("Observation date", format="YYYY-MM-DD"),
            RADIATION: st.column_config.NumberColumn(
                "Solar radiation (kWh/m²/day)", min_value=0.0, format="%.3f"
            ),
            "T2M": st.column_config.NumberColumn("Temperature (°C)", format="%.2f"),
            "RH2M": st.column_config.NumberColumn(
                "Relative humidity (%)", min_value=0.0, max_value=100.0, format="%.2f"
            ),
        },
    )
    if _input_snapshot(edited) != _input_snapshot(editor_before_change):
        st.session_state[OBSERVATION_ORIGIN_KEY] = "user_edited"
    st.session_state[OBSERVATION_EDITOR_DATA_KEY] = edited.copy()
    observation_origin = str(st.session_state[OBSERVATION_ORIGIN_KEY])
    st.caption(_observation_origin_caption(observation_origin))
    if st.session_state.get(GUIDED_EXAMPLE_KEY):
        st.info(
            "Guided value changed. Click **Run fresh Ridge prediction** to calculate "
            "the new result."
        )
    assumptions = {
        "demand": float(demand),
        "capacity": float(capacity),
        "buffer": float(buffer),
        "area": float(area),
        "efficiency": float(efficiency),
        "derating": float(derating),
    }
    observations_snapshot = _input_snapshot(edited)
    model_identity = {
        "model_id": model.model_id,
        "artifact_sha256": model.artifact_sha256,
        "training_rows": model.training_rows,
        "training_start": model.training_start.isoformat(),
        "training_end": model.training_end.isoformat(),
    }
    fingerprint_assumptions = {key: _snapshot_value(value) for key, value in assumptions.items()}
    fingerprint = _prediction_fingerprint(
        observations_snapshot, fingerprint_assumptions, model_identity, observation_origin
    )
    if st.button("Run fresh Ridge prediction", type="primary"):
        try:
            observations = _edited_window_frame(edited)
            result = predict_next_day(model, observations)
            forecasts = [
                result.radiation_kwh_m2_day,
                result.persistence_kwh_m2_day,
                result.seasonal_mean_kwh_m2_day,
            ]
            plans = plan_daily_reserve(
                forecasts,
                demand_kwh=demand,
                capacity_kwh=capacity,
                buffer_kwh=buffer,
                panel_area_m2=area,
                efficiency=efficiency,
                derating=derating,
            )
        except (TypeError, ValueError) as error:
            st.error(f"Cannot run prediction: {error}")
        else:
            st.session_state[PREDICTION_STATE_KEY] = {
                "fingerprint": fingerprint,
                "observations": observations_snapshot,
                "generated_at_utc": generated_at_utc_now(),
                "package_version": result.package_version,
                "package_source_identity": result.package_source_identity,
                "entrypoint_identity": entrypoint_source_identity(
                    "streamlit_ui", Path(__file__).read_bytes()
                ),
                "observation_origin": observation_origin,
                "assumptions": assumptions,
                "model_identity": model_identity,
                "decision_selection": decision_selection,
                "prediction": {
                    "issue_date": result.issue_date.isoformat(),
                    "target_date": result.target_date.isoformat(),
                    "radiation_kwh_m2_day": float(result.radiation_kwh_m2_day),
                    "persistence_kwh_m2_day": float(result.persistence_kwh_m2_day),
                    "seasonal_mean_kwh_m2_day": float(result.seasonal_mean_kwh_m2_day),
                    "model_id": result.model_id,
                    "features": result.features,
                    "warnings": list(result.warnings),
                },
                "reserve_plans": _json_records(plans),
            }

    saved = st.session_state.get(PREDICTION_STATE_KEY)
    if saved is None:
        st.info("Edit or confirm the seven rows, then run a fresh model inference.")
        st.caption(
            f"Loaded model: {model.model_id} · training target dates "
            f"{model.training_start} through {model.training_end} · "
            f"training rows {model.training_rows} · SHA-256 {model.artifact_sha256[:12]}…"
        )
    else:
        result_status = "fresh"
        if saved["fingerprint"] != fingerprint:
            result_status = "stale"
            st.warning(
                "Saved prediction is stale. Edited observations, reserve assumptions, or model "
                "identity differ from the values used for this result. Run a fresh prediction "
                "to replace it."
            )
        _show_saved_prediction(saved, result_status=result_status)


def main(root: Path | None = None) -> None:
    st.set_page_config(page_title="SunReserve", page_icon="☀️")
    project_root = (
        root or Path(os.environ.get("SUNRESERVE_ROOT", Path(__file__).resolve().parent))
    ).resolve()
    code_root = Path(__file__).resolve().parent
    mode = st.sidebar.radio(
        "SunReserve mode",
        options=("Next-day AI prediction", "Historical replay", "Frozen 2025 results"),
        index=0,
    )
    if mode == "Next-day AI prediction":
        experiment_id = _select_validation_experiment(project_root)
        _show_prediction(project_root, experiment_id=experiment_id)
        return
    if mode == "Frozen 2025 results":
        _show_frozen_2025(project_root)
        return
    experiment_id = _select_validation_experiment(project_root)
    try:
        artifacts = _cached_artifacts(
            str(project_root),
            str(code_root),
            artifact_signature(project_root, code_root=code_root, experiment_id=experiment_id),
            experiment_id,
        )
    except ArtifactError as error:
        _show_load_failure(error)
        return
    _show_replay(artifacts)


if __name__ == "__main__":
    main()
