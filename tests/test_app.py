from __future__ import annotations

import json
import shutil
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import app
import numpy as np
import pandas as pd
import pytest

from sunreserve.features import build_next_day_features
from sunreserve.inference import PredictionResult, RidgeModelBundle

ROOT = Path(__file__).resolve().parents[1]


class FakeUI:
    """Small Streamlit surface for deterministic page interaction checks."""

    class ColumnConfig:
        DateColumn = staticmethod(lambda *args, **kwargs: None)
        NumberColumn = staticmethod(lambda *args, **kwargs: None)

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.outputs: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.sidebar = self
        self.column_config = self.ColumnConfig()
        self.edited = sample_window()
        self.input_overrides: dict[str, float] = {}
        self.buttons_pressed: set[str] = set()
        self.uploaded_file: Any = None
        self.mode = "Next-day AI prediction"

    def __enter__(self) -> FakeUI:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        def emit(*args: Any, **kwargs: Any) -> Any:
            self.outputs.append((name, args, kwargs))
            if name == "number_input":
                return self.input_overrides.get(args[0], kwargs["value"])
            if name == "data_editor":
                if self.session_state.get(app.OBSERVATION_EDITOR_VERSION_KEY, 0) > 0:
                    return self.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
                return self.edited
            if name == "button":
                label = str(args[0]) if args else ""
                clicked = label in self.buttons_pressed
                self.buttons_pressed.discard(label)
                return clicked
            if name == "file_uploader":
                return self.uploaded_file
            if name == "radio":
                return self.mode
            if name == "columns":
                count = args[0] if isinstance(args[0], int) else len(args[0])
                return [self] * count
            if name in {"expander", "sidebar"}:
                return self
            return None

        return emit

    def values(self, name: str) -> list[str]:
        return [str(args[0]) for output, args, _ in self.outputs if output == name and args]


def sample_window() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-12-25", periods=7).date,
            "ALLSKY_SFC_SW_DWN": [2.0, 3.0, 4.5, 3.5, 5.0, 6.0, 7.0],
            "T2M": [20.0, 20.5, 21.0, 22.0, 22.5, 23.0, 23.5],
            "RH2M": [55.0, 54.0, 52.0, 50.0, 48.0, 46.0, 45.0],
        }
    )


def fake_model() -> RidgeModelBundle:
    return RidgeModelBundle(
        model_id="sunreserve.test.ridge.v1",
        feature_columns=(),
        imputer_statistics=np.array([]),
        scaler_mean=np.array([]),
        scaler_scale=np.array([]),
        coefficients=np.array([]),
        intercept=0.0,
        monthly_means={1: 5.0},
        training_global_mean=5.0,
        training_rows=100,
        training_start=date(2020, 1, 1),
        training_end=date(2021, 12, 31),
        artifact_sha256="a" * 64,
    )


def fake_prediction(model: RidgeModelBundle, observations: pd.DataFrame) -> PredictionResult:
    build_next_day_features(observations)
    issue_date = observations.index[-1].date()
    return PredictionResult(
        issue_date=issue_date,
        target_date=issue_date + pd.Timedelta(days=1),
        radiation_kwh_m2_day=6.25,
        persistence_kwh_m2_day=7.0,
        seasonal_mean_kwh_m2_day=5.0,
        model_id=model.model_id,
        features={"example": 1.0},
    )


@pytest.fixture
def prediction_ui(monkeypatch: pytest.MonkeyPatch) -> FakeUI:
    ui = FakeUI()
    model = fake_model()
    monkeypatch.setattr(app, "st", ui)
    monkeypatch.setattr(app, "load_project_model", lambda _: (model, {}))
    monkeypatch.setattr(app, "_sample_observation_window", lambda _: sample_window())
    monkeypatch.setattr(app, "predict_next_day", fake_prediction)
    return ui


def run_prediction(ui: FakeUI) -> None:
    ui.buttons_pressed.add("Run fresh Ridge prediction")
    app._show_prediction(ROOT)


def test_prediction_persists_exact_inputs_assumptions_identity_and_outputs(
    prediction_ui: FakeUI,
) -> None:
    run_prediction(prediction_ui)
    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]

    assert saved["model_identity"]["model_id"] == "sunreserve.test.ridge.v1"
    assert len(saved["observations"]) == 7
    assert saved["observations"][-1]["ALLSKY_SFC_SW_DWN"] == 7.0
    assert saved["assumptions"]["demand"] == 15.0
    assert saved["prediction"]["radiation_kwh_m2_day"] == 6.25
    assert saved["reserve_plans"]
    json.dumps(saved, allow_nan=False)
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")

    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == saved


def test_predicted_reserve_summary_shows_capacity_sufficient_and_uncertainty(
    prediction_ui: FakeUI,
) -> None:
    prediction_ui.input_overrides["Illustrative panel area (m²)"] = 5.0
    prediction_ui.input_overrides["Assumed reserve capacity (kWh)"] = 10.0
    run_prediction(prediction_ui)

    success_text = " ".join(prediction_ui.values("success"))
    uncertainty_text = " ".join(prediction_ui.values("info"))
    assert "covers the predicted backup need" in success_text
    assert "shortage can still occur" in uncertainty_text
    summary = next(
        args[0]
        for output, args, _ in prediction_ui.outputs
        if output == "dataframe" and args and "Quantity" in getattr(args[0], "columns", [])
    )
    need = summary.loc[summary["Quantity"] == "Predicted backup need before capacity cap", "Value"]
    assert need.iloc[0] == pytest.approx(10.0)


def test_predicted_reserve_summary_warns_when_capacity_is_insufficient(
    prediction_ui: FakeUI,
) -> None:
    prediction_ui.input_overrides["Illustrative panel area (m²)"] = 5.0
    prediction_ui.input_overrides["Assumed reserve capacity (kWh)"] = 2.0
    run_prediction(prediction_ui)

    warning_text = " ".join(prediction_ui.values("warning"))
    assert "exceeds available capacity" in warning_text
    assert "predicted remaining gap is 8.00 kWh/day" in warning_text


def test_zero_capacity_is_flagged_even_when_predicted_solar_exceeds_demand(
    prediction_ui: FakeUI,
) -> None:
    prediction_ui.input_overrides["Assumed reserve capacity (kWh)"] = 0.0
    run_prediction(prediction_ui)

    warning_text = " ".join(prediction_ui.values("warning"))
    summary = next(
        args[0]
        for output, args, _ in prediction_ui.outputs
        if output == "dataframe" and args and "Quantity" in getattr(args[0], "columns", [])
    )
    solar = summary.loc[
        summary["Quantity"] == "Predicted solar electricity contribution (proxy)",
        "Value",
    ]
    remaining = summary.loc[
        summary["Quantity"] == "Predicted remaining gap after solar and reserve",
        "Value",
    ]
    assert "Reserve capacity is zero" in warning_text
    assert solar.iloc[0] == pytest.approx(20.0)
    assert remaining.iloc[0] == pytest.approx(0.0)


def test_uploaded_csv_validates_rows_and_preserves_the_current_example_on_error(
    prediction_ui: FakeUI,
) -> None:
    uploaded_rows = sample_window().astype({"RH2M": "object"})
    uploaded_rows.loc[3, "RH2M"] = "not-a-number"
    prediction_ui.uploaded_file = BytesIO(uploaded_rows.to_csv(index=False).encode("utf-8"))

    app._show_prediction(ROOT)

    assert any(
        "CSV row 4 field RH2M must be numeric" in value for value in prediction_ui.values("error")
    )
    saved_window = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert saved_window.iloc[-1][app.RADIATION] == 7.0


def test_observation_csv_round_trip_and_row_specific_validation() -> None:
    source = sample_window()
    uploaded = app._parse_observation_csv(source.to_csv(index=False).encode("utf-8"))
    assert uploaded["date"].tolist() == source["date"].tolist()
    assert uploaded[app.RADIATION].tolist() == source[app.RADIATION].tolist()

    invalid = source.copy()
    invalid.loc[2, "RH2M"] = 110
    with pytest.raises(ValueError, match=r"CSV row 3 field RH2M"):
        app._parse_observation_csv(invalid.to_csv(index=False).encode("utf-8"))


def test_csv_upload_replaces_the_editor_with_validated_rows(prediction_ui: FakeUI) -> None:
    imported = sample_window()
    imported.loc[6, app.RADIATION] = 9.5
    prediction_ui.uploaded_file = BytesIO(imported.to_csv(index=False).encode("utf-8"))

    app._show_prediction(ROOT)

    saved_window = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert saved_window.iloc[-1][app.RADIATION] == 9.5
    assert any("Imported seven observations" in value for value in prediction_ui.values("success"))


def test_reset_clears_uploaded_file_and_allows_same_csv_to_be_reimported(
    prediction_ui: FakeUI,
) -> None:
    imported = sample_window()
    imported.loc[6, app.RADIATION] = 9.5
    prediction_ui.uploaded_file = BytesIO(imported.to_csv(index=False).encode("utf-8"))

    app._show_prediction(ROOT)
    assert (
        prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 9.5
    )

    prediction_ui.buttons_pressed.add("Reset to historical example")
    app._show_prediction(ROOT)
    assert (
        prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 7.0
    )
    assert prediction_ui.session_state[app.CSV_UPLOAD_VERSION_KEY] == 1
    assert prediction_ui.session_state[app.CSV_UPLOAD_PROCESSED_KEY] is None

    app._show_prediction(ROOT)
    assert (
        prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 9.5
    )


def test_reset_restores_the_historical_example(prediction_ui: FakeUI) -> None:
    prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY] = sample_window().assign(
        ALLSKY_SFC_SW_DWN=99.0
    )
    prediction_ui.buttons_pressed.add("Reset to historical example")

    app._show_prediction(ROOT)

    restored = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert restored[app.RADIATION].tolist() == sample_window()[app.RADIATION].tolist()


def test_guided_example_changes_one_input_without_running_inference(prediction_ui: FakeUI) -> None:
    prediction_ui.buttons_pressed.add("Apply the guided input change")

    app._show_prediction(ROOT)

    changed = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert changed.iloc[-1][app.RADIATION] == 8.0
    assert app.PREDICTION_STATE_KEY not in prediction_ui.session_state


def test_downloaded_result_contains_displayed_outputs_inputs_and_model(
    prediction_ui: FakeUI,
) -> None:
    run_prediction(prediction_ui)

    button = next(
        (args, kwargs)
        for output, args, kwargs in prediction_ui.outputs
        if output == "download_button"
        and args
        and "complete prediction result" in str(args[0]).lower()
    )
    payload = json.loads(button[1]["data"])
    summary = next(
        args[0]
        for output, args, _ in prediction_ui.outputs
        if output == "dataframe" and args and "Quantity" in getattr(args[0], "columns", [])
    )
    exported_ridge = payload["reserve_outputs"][0]
    assert summary.columns.tolist() == ["Quantity", "Value", "Units"]
    assert summary.iloc[0]["Quantity"] == "Assumed daily demand"
    assert "Predicted value" not in summary.columns
    assert (
        payload["observations"]
        == prediction_ui.session_state[app.PREDICTION_STATE_KEY]["observations"]
    )
    assert payload["model_identity"]["model_id"] == "sunreserve.test.ridge.v1"
    displayed_outputs = {
        "Assumed daily demand": "demand_kwh",
        "Predicted solar electricity contribution (proxy)": "predicted_generation_proxy_kwh_day",
        "Predicted backup need before capacity cap": "backup_need_before_capacity_cap_kwh",
        "Predicted planned reserve after capacity cap": "prepared_reserve_kwh",
        "Predicted remaining gap after solar and reserve": "predicted_remaining_gap_kwh",
    }
    for label, field in displayed_outputs.items():
        displayed_value = summary.loc[summary["Quantity"] == label, "Value"].iloc[0]
        assert exported_ridge[field] == pytest.approx(displayed_value)


def test_changed_assumption_marks_saved_prediction_stale(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)
    prediction_ui.input_overrides["Assumed daily demand (kWh)"] = 18.0
    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)

    assert any("stale" in value.lower() for value in prediction_ui.values("warning"))
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")


def test_changed_observation_marks_saved_prediction_stale(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)
    prediction_ui.edited.loc[6, "ALLSKY_SFC_SW_DWN"] = 8.0
    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)

    assert any("stale" in value.lower() for value in prediction_ui.values("warning"))


def test_rerun_replaces_saved_snapshot_and_clears_stale(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)
    prediction_ui.input_overrides["Assumed daily demand (kWh)"] = 18.0
    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)
    assert any("stale" in value.lower() for value in prediction_ui.values("warning"))

    prediction_ui.outputs.clear()
    run_prediction(prediction_ui)
    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    assert saved["assumptions"]["demand"] == 18.0
    assert not any("stale" in value.lower() for value in prediction_ui.values("warning"))


def test_invalid_observation_keeps_previous_result_and_reports_error(
    prediction_ui: FakeUI,
) -> None:
    run_prediction(prediction_ui)
    previous = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    prediction_ui.edited.loc[3, "ALLSKY_SFC_SW_DWN"] = -1.0
    prediction_ui.outputs.clear()
    run_prediction(prediction_ui)

    assert any("Cannot run prediction" in value for value in prediction_ui.values("error"))
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == previous
    assert any("stale" in value.lower() for value in prediction_ui.values("warning"))


def test_mode_switch_preserves_saved_prediction(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)
    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    prediction_ui.mode = "Frozen 2025 results"
    prediction_ui.outputs.clear()
    monkeypatch_main = app._show_frozen_2025
    app._show_frozen_2025 = lambda _: prediction_ui.outputs.append(("frozen", (), {}))
    try:
        app.main(ROOT)
    finally:
        app._show_frozen_2025 = monkeypatch_main

    assert any(output == "frozen" for output, _, _ in prediction_ui.outputs)
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == saved


def test_frozen_2025_view_shows_baseline_comparison_and_tradeoff(monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)
    app._show_frozen_2025(ROOT)

    assert "Frozen 2025 results" in ui.values("title")
    assert "Frozen forecast accuracy and baseline comparison" in ui.values("subheader")
    assert "Simulated reserve trade-off · primary frozen scenario" in ui.values("subheader")
    markdown = " ".join(ui.values("markdown"))
    assert "higher than persistence" in markdown
    assert len([1 for output, _, _ in ui.outputs if output == "dataframe"]) == 2


def test_frozen_2025_view_reports_missing_artifacts(tmp_path: Path, monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)
    app._show_frozen_2025(tmp_path)

    assert any("not available" in value.lower() for value in ui.values("error"))


def test_frozen_2025_loader_rejects_changed_frozen_file(tmp_path: Path) -> None:
    source = ROOT / "artifacts" / "runs" / "frozen-test-2025"
    destination = tmp_path / "artifacts" / "runs" / "frozen-test-2025"
    shutil.copytree(source, destination)
    predictions = destination / "predictions.csv"
    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(app.Frozen2025ArtifactsError, match="checksum"):
        app._load_frozen_2025(tmp_path)
