from __future__ import annotations

import json
import shutil
from datetime import date
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
        self.click_run = False
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
                return self.edited
            if name == "button":
                clicked, self.click_run = self.click_run, False
                return clicked
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
    ui.click_run = True
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
