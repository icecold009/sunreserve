"""Fresh trained-model inference and cached historical replay for SunReserve."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

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
from sunreserve.presentation import (
    ArtifactError,
    DemoArtifacts,
    artifact_signature,
    failure_examples,
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
OBSERVATION_COLUMNS = ["date", RADIATION, "T2M", "RH2M"]
FROZEN_2025_FILES = ("predictions.csv", "decision_metrics.csv", "forecast_metrics.json")


class Frozen2025ArtifactsError(RuntimeError):
    """Raised when the committed frozen 2025 result artifacts cannot be verified."""


def _load_frozen_2025(project_root: Path) -> dict[str, object]:
    """Read the frozen result files and verify their manifest-bound checksums."""
    run_dir = project_root / "artifacts" / "runs" / "frozen-test-2025"
    manifest_path = run_dir / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_bytes = {name: (run_dir / name).read_bytes() for name in FROZEN_2025_FILES}
    except (OSError, json.JSONDecodeError) as error:
        raise Frozen2025ArtifactsError(f"Frozen 2025 results are not available: {error}") from error
    checksums = manifest.get("artifact_sha256", {})
    for name, payload in file_bytes.items():
        expected = checksums.get(name)
        actual = hashlib.sha256(payload).hexdigest()
        if not expected or actual != expected:
            raise Frozen2025ArtifactsError(f"Frozen 2025 artifact checksum mismatch: {name}")
    try:
        return {
            "manifest": manifest,
            "predictions": pd.read_csv(run_dir / "predictions.csv", parse_dates=["target_date"]),
            "decisions": pd.read_csv(run_dir / "decision_metrics.csv"),
            "forecast_metrics": json.loads(file_bytes["forecast_metrics.json"].decode("utf-8")),
        }
    except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        raise Frozen2025ArtifactsError(f"Frozen 2025 results cannot be read: {error}") from error


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
    st.warning(
        "Frozen historical holdout · Applied the development-frozen model and baselines to "
        "2025 without refitting or selection on these dates. This is descriptive research, "
        "not an operational forecast or measured facility performance."
    )
    st.caption(
        f"{len(predictions)} common targets · {predictions['target_date'].min():%Y-%m-%d} to "
        f"{predictions['target_date'].max():%Y-%m-%d} · target: NASA POWER model/satellite-derived "
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
        "Assumptions: 15 kWh/day demand, 15 kWh reserve capacity, 20 m² panel area, "
        "20% conversion efficiency, 0.80 derating, and no buffer."
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
    st.markdown(
        f"Ridge radiation MAE is **{metrics['ridge']['mae']:.3f}**, compared with "
        f"**{metrics['persistence']['mae']:.3f}** for persistence and "
        f"**{metrics['seasonal_mean']['mae']:.3f}** for the training-only seasonal mean. "
        f"Ridge's simulated shortfall is **{ridge_shortfall:.3f} kWh/day**, higher than "
        f"persistence at **{persistence_shortfall:.3f} kWh/day**. The equal-weight "
        "shortfall-plus-excess score is hypothetical; it is not a measured cost or "
        "reliability target."
    )
    st.line_chart(
        predictions.set_index("target_date")[
            [
                "target_radiation_kwh_m2_day",
                "ridge_prediction_kwh_m2_day",
                "persistence_prediction_kwh_m2_day",
                "seasonal_mean_prediction_kwh_m2_day",
            ]
        ],
        y_label="Solar radiation (kWh/m²/day)",
    )
    st.caption(
        f"Frozen run {manifest.get('run_id', 'frozen-test-2025')} · source artifact files "
        "were checksum-verified before display."
    )


def _format_kwh_day(value: float) -> str:
    """Keep simulated-energy metric values compact enough for a narrow Streamlit card."""
    return f"{value:.2f} kWh/day"


@st.cache_data(show_spinner=False)
def _cached_artifacts(root_text: str, code_root_text: str, signature: tuple) -> DemoArtifacts:
    """Cache local reads by file metadata; never fetch, fit, or rewrite artifacts."""
    del signature
    return load_demo_artifacts(Path(root_text), code_root=Path(code_root_text))


def _show_load_failure(error: ArtifactError) -> None:
    st.title("SunReserve historical replay")
    st.warning(f"Cached validation data is {error.category}.")
    st.write(str(error))
    st.markdown("To create or refresh local validation outputs, run these commands in the project:")
    st.code(
        "python -m sunreserve fetch\npython -m sunreserve prepare\npython -m sunreserve validate",
        language="bash",
    )
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
        "NASA POWER daily solar radiation is model/satellite-derived and reported in "
        "kWh/m²/day. Electricity, prepared reserve, shortfall, and excess are hypothetical "
        "simulated kWh/day quantities."
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
        "NASA POWER solar radiation",
        f"{selected_forecast[RADIATION_COLUMN]:.2f} kWh/m²/day",
    )
    metric_columns[1].metric(
        "Predicted solar radiation",
        "Not forecast-based" if forecast_value is None else f"{forecast_value:.2f} kWh/m²/day",
    )
    reserve_columns = st.columns(2)
    reserve_columns[0].metric(
        "Prepared reserve",
        _format_kwh_day(decision["prepared_reserve_kwh"]),
    )
    reserve_columns[1].metric(
        "Simulated actual electricity proxy",
        _format_kwh_day(decision["actual_generation_proxy_kwh_day"]),
        help="Radiation × assumed panel area × efficiency × derating; not measured generation.",
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
        "Simulated shortfall",
        _format_kwh_day(decision["simulated_shortfall_kwh"]),
        help="Includes reserve-capacity limits and forecast-dependent reserve preparation.",
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

    rows: list[dict[str, object]] = []
    previous_date: date | None = None
    for position, record in enumerate(uploaded.to_dict(orient="records"), start=1):
        raw_date = record["date"]
        if pd.isna(raw_date) or not isinstance(raw_date, str):
            raise ValueError(f"CSV row {position} is missing its date; use YYYY-MM-DD.")
        try:
            observation_date = date.fromisoformat(raw_date)
        except ValueError as error:
            raise ValueError(f"CSV row {position} has an invalid date; use YYYY-MM-DD.") from error
        if observation_date.isoformat() != raw_date:
            raise ValueError(f"CSV row {position} date must use YYYY-MM-DD.")
        if observation_date > date.today():
            raise ValueError(f"CSV row {position} date is in the future.")
        if previous_date is not None and observation_date != previous_date + timedelta(days=1):
            raise ValueError(
                f"CSV row {position} must be the next consecutive calendar day after row "
                f"{position - 1}."
            )
        converted: dict[str, object] = {"date": observation_date}
        for column in OBSERVATION_COLUMNS[1:]:
            raw_value = record[column]
            if pd.isna(raw_value):
                raise ValueError(f"CSV row {position} field {column} is blank.")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV row {position} field {column} must be numeric.") from error
            if not math.isfinite(value):
                raise ValueError(f"CSV row {position} field {column} must be finite.")
            if column == RADIATION and value < 0:
                raise ValueError(f"CSV row {position} field {column} must be nonnegative.")
            if column == "RH2M" and not 0 <= value <= 100:
                raise ValueError(
                    f"CSV row {position} field RH2M must be between 0 and 100 percent."
                )
            converted[column] = value
        rows.append(converted)
        previous_date = observation_date
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)


def _edited_window_frame(edited: pd.DataFrame) -> pd.DataFrame:
    """Convert Streamlit editor rows to the strict inference input contract."""
    records = edited.to_dict(orient="records")
    for position, record in enumerate(records, start=1):
        raw_date = record.get("date")
        if raw_date is None or pd.isna(raw_date):
            raise ValueError(f"observation {position} is missing its date")
        try:
            record["date"] = pd.Timestamp(raw_date).date().isoformat()
        except (TypeError, ValueError) as error:
            raise ValueError(f"observation {position} has an invalid date") from error
    return observation_frame_from_records(records)


def _snapshot_value(value: object) -> object:
    """Convert editor values into stable JSON-safe values without hiding invalid edits."""
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
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
    assumptions: dict[str, float],
    model_identity: dict[str, object],
) -> str:
    payload = {"observations": observations, "assumptions": assumptions, "model": model_identity}
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


def _prediction_export_payload(saved: dict[str, object]) -> dict[str, object]:
    """Build a portable result containing the exact inputs, assumptions, model, and outputs."""
    plans = _prediction_decision_table(saved)
    return {
        "format": "sunreserve-prediction-v1",
        "interpretation": (
            "Forecast and reserve quantities are hypothetical predictions; no target-day "
            "outcome or facility performance is measured."
        ),
        "observations": saved["observations"],
        "assumptions": saved["assumptions"],
        "model_identity": saved["model_identity"],
        "prediction": saved["prediction"],
        "reserve_outputs": _json_records(plans),
    }


def _show_saved_prediction(saved: dict[str, object]) -> None:
    prediction = saved["prediction"]
    plans = _prediction_decision_table(saved)
    forecasts = [
        prediction["radiation_kwh_m2_day"],
        prediction["persistence_kwh_m2_day"],
        prediction["seasonal_mean_kwh_m2_day"],
    ]
    st.subheader(f"Fresh prediction for {prediction['target_date']}")
    st.metric("Ridge predicted solar radiation", f"{forecasts[0]:.2f} kWh/m²/day")
    ridge_plan = plans.iloc[0]
    st.subheader("Predicted reserve summary · Ridge")
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
    export_data = json.dumps(
        _prediction_export_payload(saved), ensure_ascii=False, indent=2, allow_nan=False
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


def _show_prediction(project_root: Path) -> None:
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
    try:
        model, decision_selection = load_project_model(project_root)
    except ModelArtifactError as error:
        st.error(f"Trained model unavailable: {error}")
        st.code(
            "python -m sunreserve fetch\n"
            "python -m sunreserve prepare\n"
            "python -m sunreserve validate"
        )
        st.caption("Prediction mode requires the fitted model produced by validation.")
        return

    config = ExperimentConfig()
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
    if OBSERVATION_EDITOR_VERSION_KEY not in st.session_state:
        st.session_state[OBSERVATION_EDITOR_VERSION_KEY] = 0
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
                st.session_state[CSV_UPLOAD_STATUS_KEY] = (
                    uploaded_digest,
                    "Imported seven observations. Review the rows, then run prediction when ready.",
                    True,
                )
            st.session_state[CSV_UPLOAD_PROCESSED_KEY] = uploaded_digest
        status = st.session_state.get(CSV_UPLOAD_STATUS_KEY)
        if status and status[0] == uploaded_digest:
            (st.success if status[2] else st.error)(status[1])

    reset_column, guide_column = st.columns(2)
    if reset_column.button("Reset to historical example", key="sunreserve_reset_example"):
        _set_observation_editor_data(editor)
        st.session_state[CSV_UPLOAD_VERSION_KEY] = upload_version + 1
        st.session_state[CSV_UPLOAD_PROCESSED_KEY] = None
        st.session_state[CSV_UPLOAD_STATUS_KEY] = None
        st.session_state[GUIDED_EXAMPLE_KEY] = False
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
    st.session_state[OBSERVATION_EDITOR_DATA_KEY] = edited.copy()
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
    fingerprint = _prediction_fingerprint(observations_snapshot, assumptions, model_identity)
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
        if saved["fingerprint"] != fingerprint:
            st.warning(
                "Saved prediction is stale. Edited observations, reserve assumptions, or model "
                "identity differ from the values used for this result. Run a fresh prediction "
                "to replace it."
            )
        _show_saved_prediction(saved)


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
        _show_prediction(project_root)
        return
    if mode == "Frozen 2025 results":
        _show_frozen_2025(project_root)
        return
    try:
        artifacts = _cached_artifacts(
            str(project_root),
            str(code_root),
            artifact_signature(project_root, code_root=code_root),
        )
    except ArtifactError as error:
        _show_load_failure(error)
        return
    _show_replay(artifacts)


if __name__ == "__main__":
    main()
