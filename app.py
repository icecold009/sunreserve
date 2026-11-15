"""Fresh trained-model inference and cached historical replay for SunReserve."""

from __future__ import annotations

import os
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
    st.caption(
        "A prefilled local example, when available, is historical and may not have been "
        "available on its issue date. Replace it with verified recent observations before use."
    )
    editor = _sample_observation_window(paths)
    edited = st.data_editor(
        editor,
        key="sunreserve_prediction_observations",
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
    if not st.button("Run fresh Ridge prediction", type="primary"):
        st.info("Edit or confirm the seven rows, then run a fresh model inference.")
        st.caption(
            f"Loaded model: {model.model_id} · training target dates "
            f"{model.training_start} through {model.training_end} · "
            f"training rows {model.training_rows} · SHA-256 {model.artifact_sha256[:12]}…"
        )
        return

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
        return

    st.subheader(f"Fresh prediction for {result.target_date:%Y-%m-%d}")
    st.metric(
        "Ridge predicted solar radiation",
        f"{result.radiation_kwh_m2_day:.2f} kWh/m²/day",
    )
    st.metric(
        "Ridge planned reserve",
        f"{plans.loc[0, 'prepared_reserve_kwh']:.2f} kWh",
    )
    st.caption(
        f"Model {result.model_id}; trained on {model.training_rows} training rows through "
        f"{model.training_end}. Model SHA-256: {model.artifact_sha256}."
    )
    if decision_selection.get("selected_model"):
        st.caption(
            "Development-only reserve objective selected "
            f"{decision_selection['selected_model']} on the 2022 validation period. "
            "This is a research ranking, not an operational reliability guarantee."
        )
    comparison = pd.DataFrame(
        {
            "Approach": ["Trained Ridge model", "Persistence baseline", "Seasonal-mean baseline"],
            "Predicted radiation (kWh/m²/day)": forecasts,
            "Predicted electricity proxy (kWh/day)": plans[
                "predicted_generation_proxy_kwh_day"
            ].to_list(),
            "Prepared reserve (kWh)": plans["prepared_reserve_kwh"].to_list(),
        }
    )
    st.subheader("Fresh inference and baseline reserve comparison")
    st.dataframe(comparison, hide_index=True, width="stretch")
    chart_columns = st.columns(2)
    chart_columns[0].bar_chart(
        comparison.set_index("Approach")["Predicted radiation (kWh/m²/day)"],
        y_label="Solar radiation (kWh/m²/day)",
    )
    chart_columns[1].bar_chart(
        comparison.set_index("Approach")["Prepared reserve (kWh)"],
        y_label="Prepared reserve (kWh)",
    )
    st.info(
        "The target day's actual radiation is not yet supplied, so realized shortfall and "
        "excess prepared reserve are not calculated in prediction mode. No forecast-data "
        "availability or facility performance is established."
    )
    with st.expander("Feature and model details"):
        st.write(
            f"Issue date: {result.issue_date} · target date: {result.target_date} · "
            f"model: {result.model_id} · training rows: {model.training_rows}."
        )
        st.write("Feature definitions match the historical pipeline:")
        st.json(result.features, expanded=False)
        st.markdown(
            "Training-only median imputation, standard scaling, and Ridge coefficients are "
            "loaded from the versioned numeric model bundle. Input window values are not "
            "reused cached predictions."
        )
    st.caption(
        f"Assumptions: demand {demand:g} kWh/day, reserve capacity {capacity:g} kWh, "
        f"buffer {buffer:g} kWh, area {area:g} m², efficiency {efficiency:g}, "
        f"derating {derating:g}. All are hypothetical, not measured facility values."
    )


def main(root: Path | None = None) -> None:
    st.set_page_config(page_title="SunReserve", page_icon="☀️")
    project_root = (
        root or Path(os.environ.get("SUNRESERVE_ROOT", Path(__file__).resolve().parent))
    ).resolve()
    code_root = Path(__file__).resolve().parent
    mode = st.sidebar.radio(
        "SunReserve mode",
        options=("Next-day AI prediction", "Historical replay"),
        index=0,
    )
    if mode == "Next-day AI prediction":
        _show_prediction(project_root)
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
