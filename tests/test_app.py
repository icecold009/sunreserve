from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import app
import numpy as np
import pandas as pd
import pytest

from sunreserve.config import ExperimentConfig
from sunreserve.features import build_next_day_features
from sunreserve.inference import ModelArtifactError, PredictionResult, RidgeModelBundle
from sunreserve.prediction_export import PREDICTION_EXPORT_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def _copy_frozen_2025(tmp_path: Path) -> Path:
    source = ROOT / "artifacts" / "runs" / "frozen-test-2025"
    destination = tmp_path / "artifacts" / "runs" / "frozen-test-2025"
    shutil.copytree(source, destination)
    return tmp_path


def _replace_frozen_output(root: Path, filename: str, payload: bytes) -> None:
    run_dir = root / "artifacts" / "runs" / "frozen-test-2025"
    (run_dir / filename).write_bytes(payload)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][filename] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


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
        self.artifact_source: str | None = "shipped"

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
                callback = kwargs.get("on_click")
                if clicked and callable(callback):
                    callback_args = kwargs.get("args") or ()
                    callback_kwargs = kwargs.get("kwargs") or {}
                    callback(*callback_args, **callback_kwargs)
                return clicked
            if name == "file_uploader":
                return self.uploaded_file
            if name == "radio":
                return self.mode
            if name == "selectbox":
                return self.artifact_source
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
        package_version="0.1.0",
        package_source_identity={
            "identity_type": "installed_package_python_sources",
            "algorithm": "sha256",
            "digest": "b" * 64,
            "file_count": 1,
        },
        warnings=(app.OBSERVATION_POLICY_NOTE,),
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
    assert saved["prediction"]["warnings"] == [app.OBSERVATION_POLICY_NOTE]
    assert saved["reserve_plans"]
    json.dumps(saved, allow_nan=False)
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")
    assert any(app.OBSERVATION_POLICY_NOTE in value for value in prediction_ui.values("warning"))

    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == saved


@pytest.mark.parametrize(
    ("value", "marker"),
    [(float("inf"), "positive_infinity"), (float("-inf"), "negative_infinity")],
)
def test_nonfinite_editor_value_is_stale_and_does_not_replace_last_result(
    prediction_ui: FakeUI, value: float, marker: str
) -> None:
    run_prediction(prediction_ui)
    original = prediction_ui.session_state[app.PREDICTION_STATE_KEY].copy()
    prediction_ui.edited.loc[6, app.RADIATION] = value

    snapshot = app._input_snapshot(prediction_ui.edited)
    assert snapshot[6][app.RADIATION] == {"$nonfinite_float": marker}
    json.dumps(snapshot, allow_nan=False)

    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)
    assert any("Saved prediction is stale" in value for value in prediction_ui.values("warning"))
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == original

    prediction_ui.outputs.clear()
    prediction_ui.buttons_pressed.add("Run fresh Ridge prediction")
    app._show_prediction(ROOT)
    assert any(
        "field ALLSKY_SFC_SW_DWN must contain finite values" in value
        for value in prediction_ui.values("error")
    )
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == original


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_sidebar_assumption_is_stale_and_does_not_replace_last_result(
    prediction_ui: FakeUI, value: float
) -> None:
    run_prediction(prediction_ui)
    original = prediction_ui.session_state[app.PREDICTION_STATE_KEY].copy()
    prediction_ui.input_overrides["Assumed daily demand (kWh)"] = value

    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)

    assert any("Saved prediction is stale" in value for value in prediction_ui.values("warning"))
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == original

    prediction_ui.outputs.clear()
    prediction_ui.buttons_pressed.add("Run fresh Ridge prediction")
    app._show_prediction(ROOT)
    assert any(
        "Cannot run prediction: demand_kwh" in value for value in prediction_ui.values("error")
    )
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == original


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

    invalid = source.copy()
    invalid.loc[2, "T2M"] = -999
    with pytest.raises(ValueError, match=r"CSV row 3 field T2M.*NASA POWER -999"):
        app._parse_observation_csv(invalid.to_csv(index=False).encode("utf-8"))


def test_editor_uses_shared_physical_contract() -> None:
    invalid = sample_window()
    invalid.loc[2, "T2M"] = -999

    with pytest.raises(ValueError, match=r"observation 3 field T2M.*NASA POWER -999"):
        app._edited_window_frame(invalid)


def test_editor_accepts_midnight_dates_and_rejects_time_or_timezone() -> None:
    midnight = sample_window()
    midnight["date"] = pd.date_range("2024-12-25", periods=7, freq="D")
    assert len(app._edited_window_frame(midnight)) == 7

    subdaily = midnight.copy()
    subdaily.loc[2, "date"] = pd.Timestamp("2024-12-27 12:00")
    with pytest.raises(
        ValueError, match=r"observation 3 date must be a timezone-naive calendar date"
    ):
        app._edited_window_frame(subdaily)

    timezone_aware = midnight.copy()
    timezone_aware["date"] = timezone_aware["date"].astype(object)
    timezone_aware.loc[2, "date"] = pd.Timestamp("2024-12-27", tz="Asia/Kolkata")
    with pytest.raises(
        ValueError, match=r"observation 3 date must be a timezone-naive calendar date"
    ):
        app._edited_window_frame(timezone_aware)


def test_csv_upload_replaces_the_editor_with_validated_rows(prediction_ui: FakeUI) -> None:
    imported = sample_window()
    imported.loc[6, app.RADIATION] = 9.5
    prediction_ui.uploaded_file = BytesIO(imported.to_csv(index=False).encode("utf-8"))

    app._show_prediction(ROOT)

    saved_window = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert saved_window.iloc[-1][app.RADIATION] == 9.5
    assert prediction_ui.session_state[app.OBSERVATION_ORIGIN_KEY] == "csv_import"
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
    assert prediction_ui.session_state[app.OBSERVATION_ORIGIN_KEY] == "local_historical_example"


def test_guided_example_changes_one_input_without_running_inference(prediction_ui: FakeUI) -> None:
    prediction_ui.buttons_pressed.add("Apply the guided input change")

    app._show_prediction(ROOT)

    changed = prediction_ui.session_state[app.OBSERVATION_EDITOR_DATA_KEY]
    assert changed.iloc[-1][app.RADIATION] == 8.0
    assert prediction_ui.session_state[app.OBSERVATION_ORIGIN_KEY] == "guided_example"
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
    assert set(payload) == PREDICTION_EXPORT_FIELDS
    assert payload["result_status"] == "fresh"
    assert payload["issue_date"] == "2024-12-31"
    assert payload["target_date"] == "2025-01-01"
    generated_at = datetime.fromisoformat(payload["generated_at_utc"].replace("Z", "+00:00"))
    assert generated_at.utcoffset() == UTC.utcoffset(generated_at)
    assert payload["source"]["kind"] == "local_historical_example"
    assert payload["source"]["availability_status"] == "unverified"
    assert "availability" in payload["source"]["availability_note"]
    assert payload["assumptions"]["daily_demand_kwh"] == {
        "value": 15.0,
        "unit": "kWh/day",
    }
    assert payload["assumptions"]["panel_area_m2"] == {"value": 20.0, "unit": "m²"}
    assert payload["package_version"] == "0.1.0"
    assert payload["package_source_identity"]["identity_type"] == (
        "installed_package_python_sources"
    )
    assert payload["package_source_identity"]["digest"] == "b" * 64
    assert payload["entrypoint_identity"] == {
        "application_kind": "streamlit_ui",
        "algorithm": "sha256",
        "digest": hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
    }
    assert payload["model_identity"]["artifact_sha256"] == "a" * 64
    assert payload["model_identity"]["training_period"] == {
        "start_date": "2020-01-01",
        "end_date": "2021-12-31",
    }
    assert payload["forecasts"]["ridge_radiation_kwh_m2_day"] == 6.25
    assert [row["approach"] for row in payload["reserve_outputs"]] == [
        "ridge",
        "persistence",
        "seasonal_mean",
    ]
    assert str(ROOT) not in json.dumps(payload)
    assert payload["prediction"]["warnings"] == [app.OBSERVATION_POLICY_NOTE]
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


def test_older_saved_prediction_without_run_timestamp_renders_without_export(
    prediction_ui: FakeUI,
) -> None:
    run_prediction(prediction_ui)
    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    saved.pop("generated_at_utc")
    saved.pop("entrypoint_identity")
    prediction_ui.outputs.clear()

    app._show_saved_prediction(saved, result_status="stale")

    assert any(
        "predates the JSON export provenance contract" in value
        for value in prediction_ui.values("info")
    )
    assert not any(output == "download_button" for output, _, _ in prediction_ui.outputs)


def test_changed_assumption_marks_saved_prediction_stale(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)
    prediction_ui.input_overrides["Assumed daily demand (kWh)"] = 18.0
    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)

    assert any("stale" in value.lower() for value in prediction_ui.values("warning"))
    assert "Fresh prediction for 2025-01-01" in prediction_ui.values("subheader")
    button = next(
        kwargs
        for output, args, kwargs in prediction_ui.outputs
        if output == "download_button"
        and args
        and "complete prediction result" in str(args[0]).lower()
    )
    assert json.loads(button["data"])["result_status"] == "stale"


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
    assert not prediction_ui.values("selectbox")
    assert prediction_ui.session_state[app.PREDICTION_STATE_KEY] == saved


def test_experiment_selector_routes_prediction_and_replay(monkeypatch, tmp_path) -> None:
    ui = FakeUI()
    ui.artifact_source = "trial-1"
    monkeypatch.setattr(app, "st", ui)
    monkeypatch.setattr(app, "list_validation_experiments", lambda _root: ["trial-1"])
    selected = []
    monkeypatch.setattr(
        app,
        "_show_prediction",
        lambda _root, experiment_id=None: selected.append(("prediction", experiment_id)),
    )
    monkeypatch.setattr(app, "artifact_signature", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(app, "_cached_artifacts", lambda *_args: object())
    monkeypatch.setattr(
        app, "_show_replay", lambda _artifacts: selected.append(("replay", "loaded"))
    )

    app.main(tmp_path)
    ui.mode = "Historical replay"
    app.main(tmp_path)

    assert selected == [("prediction", "trial-1"), ("replay", "loaded")]
    assert ui.values("selectbox") == ["Validation artifact set", "Validation artifact set"]


def test_selected_prediction_passes_expected_context_before_location_ui(
    monkeypatch: pytest.MonkeyPatch, prediction_ui: FakeUI
) -> None:
    calls = []

    def reject_selected_context(
        _root: Path,
        *,
        experiment_id: str | None = None,
        expected_context: ExperimentConfig | None = None,
    ):
        calls.append((experiment_id, expected_context))
        raise ModelArtifactError("selected experiment context mismatch")

    monkeypatch.setattr(app, "load_project_model", reject_selected_context)

    app._show_prediction(ROOT, experiment_id="trial-1")

    assert calls == [("trial-1", ExperimentConfig())]
    assert any("Trained model unavailable" in value for value in prediction_ui.values("error"))
    assert not any("Reference location" in value for value in prediction_ui.values("markdown"))
    assert not prediction_ui.values("number_input")


def test_recovery_guidance_separates_shipped_restore_from_new_experiment(monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)

    app._show_load_failure(app.ArtifactError("missing", "validation outputs are missing"))

    commands = "\n".join(ui.values("code"))
    captions = "\n".join(ui.values("caption"))
    assert "git restore --source=HEAD -- artifacts/validation models/ridge_model.json" in commands
    assert "python -m sunreserve validate --run-id recovery-2026-11-19" in commands
    assert "artifacts/experiments/recovery-2026-11-19/" in captions
    assert "does not replace the shipped default artifacts" in captions
    assert "python -m sunreserve validate\n" not in commands


def test_frozen_2025_view_shows_baseline_comparison_and_tradeoff(monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)
    app._show_frozen_2025(ROOT)

    assert "Frozen 2025 results" in ui.values("title")
    assert "Frozen forecast accuracy and baseline comparison" in ui.values("subheader")
    assert "Simulated reserve trade-off · primary frozen scenario" in ui.values("subheader")
    assert "Frozen monthly forecast error" in ui.values("subheader")
    assert "Frozen seasonal forecast error" in ui.values("subheader")
    assert "Frozen low-solar slice" in ui.values("subheader")
    assert "Unfavorable frozen forecast examples" in ui.values("subheader")
    markdown = " ".join(ui.values("markdown"))
    assert "higher than persistence" in markdown
    captions = " ".join(ui.values("caption"))
    assert "NASA POWER Daily API" in captions
    assert "12.9716" in captions and "77.5946" in captions
    assert "UTC" in captions
    assert "Frozen scenario settings" in captions
    assert "live editable assumptions" in captions
    assert "4.5967" in captions
    assert "Cells are MAE in kWh/m²/day" in captions
    assert "MAE, RMSE, and bias are in kWh/m²/day" in captions
    frames = [args[0] for output, args, _ in ui.outputs if output == "dataframe"]
    monthly = next(frame for frame in frames if "Month" in frame.columns)
    seasonal = next(frame for frame in frames if "Season" in frame.columns)
    low_solar = next(frame for frame in frames if "RMSE" in frame.columns)
    examples = next(
        frame
        for frame in frames
        if any(str(column).startswith("Prediction minus observed") for column in frame.columns)
    )
    assert len(monthly) == 12
    assert monthly["n"].sum() == 365
    assert monthly.columns.tolist() == [
        "Month",
        "n",
        "Ridge",
        "Persist.",
        "Seasonal",
    ]
    assert len(seasonal) == 4
    assert seasonal["n"].sum() == 365
    assert seasonal.columns.tolist() == [
        "Season",
        "n",
        "Ridge",
        "Persist.",
        "Seasonal",
    ]
    assert low_solar.columns.tolist() == ["Approach", "Days", "MAE", "RMSE", "Bias"]
    assert set(low_solar["Days"]) == {124}
    assert len(examples) == 6
    assert set(examples["Example"]) == {"Largest overprediction", "Largest underprediction"}
    frozen_chart = next(args[0] for output, args, _ in ui.outputs if output == "line_chart")
    assert frozen_chart.columns.tolist() == [
        "Observed NASA POWER target",
        "Ridge prediction",
        "Persistence prediction",
        "Seasonal-mean prediction",
    ]


def test_replay_labels_2022_source_location_time_and_realized_simulation(monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)

    def choose_default(label, options, index=0, **kwargs):
        ui.outputs.append(("selectbox", (label,), {"options": options, "index": index, **kwargs}))
        return options[index]

    def choose_default_date(label, **kwargs):
        ui.outputs.append(("date_input", (label,), kwargs))
        return kwargs["value"]

    ui.selectbox = choose_default
    ui.date_input = choose_default_date
    app._show_replay(app.load_demo_artifacts(ROOT))

    warning = " ".join(ui.values("warning"))
    captions = " ".join(ui.values("caption"))
    assert "2022 validation replay" in warning
    assert "NASA POWER Daily API" in captions
    assert "12.9716" in captions and "77.5946" in captions
    assert "UTC" in captions
    assert "post-hoc" in captions.lower() and "observed radiation" in captions.lower(), ui.values(
        "caption"
    )


def test_prediction_labels_input_origin_and_reference_basis(prediction_ui: FakeUI) -> None:
    run_prediction(prediction_ui)

    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    assert saved["observation_origin"] == "local_historical_example"
    assert any(
        "local historical example" in value.lower() for value in prediction_ui.values("caption")
    )
    assert any("Bengaluru" in value and "UTC" in value for value in prediction_ui.values("caption"))

    prediction_ui.edited.loc[6, app.RADIATION] = 8.0
    prediction_ui.buttons_pressed.add("Run fresh Ridge prediction")
    prediction_ui.outputs.clear()
    app._show_prediction(ROOT)

    saved = prediction_ui.session_state[app.PREDICTION_STATE_KEY]
    assert saved["observation_origin"] == "user_edited"
    assert any(
        "user-edited observations" in value.lower() for value in prediction_ui.values("caption")
    )


def test_frozen_2025_view_reports_missing_artifacts(tmp_path: Path, monkeypatch) -> None:
    ui = FakeUI()
    monkeypatch.setattr(app, "st", ui)
    app._show_frozen_2025(tmp_path)

    assert any("not available" in value.lower() for value in ui.values("error"))


def test_frozen_2025_loader_rejects_changed_frozen_file(tmp_path: Path) -> None:
    _copy_frozen_2025(tmp_path)
    destination = tmp_path / "artifacts" / "runs" / "frozen-test-2025"
    predictions = destination / "predictions.csv"
    predictions.write_text(predictions.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(app.Frozen2025ArtifactsError, match="checksum"):
        app._load_frozen_2025(tmp_path)


def test_frozen_2025_loader_reads_and_parses_each_verified_snapshot_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _copy_frozen_2025(tmp_path)
    run_dir = root / "artifacts" / "runs" / "frozen-test-2025"
    prediction_path = run_dir / "predictions.csv"
    original_predictions = prediction_path.read_bytes()
    consumed_paths = {run_dir / "run_manifest.json"} | {
        run_dir / filename for filename in app.FROZEN_2025_FILES
    }
    read_counts: dict[str, int] = {}
    original_read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path in consumed_paths:
            read_counts[path.name] = read_counts.get(path.name, 0) + 1
            if path == prediction_path:
                path.write_bytes(b"invalid replacement after snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    try:
        result = app._load_frozen_2025(root)
        assert len(result["predictions"]) == 365
        assert result["predictions"]["target_date"].min() == pd.Timestamp("2025-01-01")
        assert read_counts == {
            "run_manifest.json": 1,
            "predictions.csv": 1,
            "decision_metrics.csv": 1,
            "forecast_metrics.json": 1,
        }
    finally:
        prediction_path.write_bytes(original_predictions)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema version"),
        ("artifact_type", "different_artifact", "identity or target period"),
        ("partition", "validation", "identity or target period"),
        ("target_period", ["2024-01-01", "2024-12-31"], "identity or target period"),
    ],
)
def test_frozen_2025_loader_rejects_manifest_identity_defects(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root = _copy_frozen_2025(tmp_path)
    manifest_path = root / "artifacts" / "runs" / "frozen-test-2025" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(app.Frozen2025ArtifactsError, match=message):
        app._load_frozen_2025(root)


def test_frozen_2025_loader_rejects_unrepresentable_manifest_number(tmp_path: Path) -> None:
    root = _copy_frozen_2025(tmp_path)
    manifest_path = root / "artifacts" / "runs" / "frozen-test-2025" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decision_scenario"]["panel_area_m2"] = 10**400
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(app.Frozen2025ArtifactsError, match="must be finite"):
        app._load_frozen_2025(root)


@pytest.mark.parametrize(
    ("defect", "filename", "message"),
    [
        ("wrong_split", "predictions.csv", "unsupported split"),
        ("issue_date", "predictions.csv", "issue dates"),
        ("duplicate_target", "predictions.csv", "sorted unique dates"),
        ("negative_prediction", "predictions.csv", "must be nonnegative"),
        ("extra_prediction_column", "predictions.csv", "unsupported"),
        ("bad_metric_units", "forecast_metrics.json", "period, or units"),
        ("nonfinite_forecast_metric", "forecast_metrics.json", "non-finite"),
        ("mismatched_forecast_metric", "forecast_metrics.json", "does not match predictions.csv"),
        ("unexpected_scenario", "decision_metrics.csv", "scenarios do not match"),
        ("unexpected_approach", "decision_metrics.csv", "incomplete approach matrix"),
        ("wrong_sample_count", "decision_metrics.csv", "invalid sample count"),
        ("decision_score", "decision_metrics.csv", "do not match their component metrics"),
    ],
)
def test_frozen_2025_loader_rejects_concrete_output_defects(
    tmp_path: Path, defect: str, filename: str, message: str
) -> None:
    root = _copy_frozen_2025(tmp_path)
    run_dir = root / "artifacts" / "runs" / "frozen-test-2025"
    output_path = run_dir / filename
    if filename.endswith(".csv"):
        frame = pd.read_csv(output_path)
        if defect == "wrong_split":
            frame.loc[0, "split"] = "validation"
        elif defect == "issue_date":
            frame.loc[0, "issue_date"] = frame.loc[0, "target_date"]
        elif defect == "duplicate_target":
            frame.loc[len(frame) - 1, "target_date"] = frame.loc[len(frame) - 2, "target_date"]
        elif defect == "negative_prediction":
            frame.loc[0, "target_radiation_kwh_m2_day"] = -1.0
        elif defect == "extra_prediction_column":
            frame["unexpected"] = 1
        elif defect == "unexpected_scenario":
            frame.loc[0, "scenario_id"] = "secondary"
        elif defect == "unexpected_approach":
            frame.loc[0, "approach"] = "always_prepare_backup"
        elif defect == "wrong_sample_count":
            frame.loc[0, "sample_count"] = 364
        elif defect == "decision_score":
            frame.loc[0, "mean_shortfall_plus_mean_excess_kwh_day"] += 1.0
        payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    else:
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
        if defect == "bad_metric_units":
            metrics["target_units"] = "MJ/m^2/day"
        elif defect == "nonfinite_forecast_metric":
            metrics["metrics"]["ridge"]["mae"] = float("inf")
        elif defect == "mismatched_forecast_metric":
            metrics["metrics"]["ridge"]["mae"] += 1.0
        payload = json.dumps(metrics).encode("utf-8")
    _replace_frozen_output(root, filename, payload)

    with pytest.raises(app.Frozen2025ArtifactsError, match=message):
        app._load_frozen_2025(root)
