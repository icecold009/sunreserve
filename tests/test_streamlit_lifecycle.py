from __future__ import annotations

import os
from datetime import date
from typing import Any

import app
import numpy as np
import pandas as pd
import pytest

from sunreserve.features import build_next_day_features
from sunreserve.inference import PredictionResult, RidgeModelBundle

pytestmark = [
    pytest.mark.streamlit,
    pytest.mark.skipif(
        os.environ.get("SUNRESERVE_RUN_STREAMLIT_TESTS") != "1",
        reason="real Streamlit lifecycle tests run in the socket-enabled CI smoke job",
    ),
]


def _sample_window() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-12-25", periods=7).date,
            "ALLSKY_SFC_SW_DWN": [2.0, 3.0, 4.5, 3.5, 5.0, 6.0, 7.0],
            "T2M": [20.0, 20.5, 21.0, 22.0, 22.5, 23.0, 23.5],
            "RH2M": [55.0, 54.0, 52.0, 50.0, 48.0, 46.0, 45.0],
        }
    )


def _fake_model() -> RidgeModelBundle:
    return RidgeModelBundle(
        model_id="sunreserve.lifecycle.ridge.v1",
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


def _fake_prediction(model: RidgeModelBundle, observations: pd.DataFrame) -> PredictionResult:
    build_next_day_features(observations)
    issue_date = observations.index[-1].date()
    radiation = float(observations["ALLSKY_SFC_SW_DWN"].iloc[-1])
    return PredictionResult(
        issue_date=issue_date,
        target_date=issue_date + pd.Timedelta(days=1),
        radiation_kwh_m2_day=radiation,
        persistence_kwh_m2_day=radiation,
        seasonal_mean_kwh_m2_day=5.0,
        model_id=model.model_id,
        features={"lifecycle_fixture": radiation},
        package_version="0.1.0",
        package_source_identity={
            "identity_type": "installed_package_python_sources",
            "algorithm": "sha256",
            "digest": "c" * 64,
            "file_count": 1,
        },
        warnings=(app.OBSERVATION_POLICY_NOTE,),
    )


def _new_app_test(monkeypatch: pytest.MonkeyPatch):
    from streamlit.testing.v1 import AppTest

    model = _fake_model()
    monkeypatch.setattr(app, "load_project_model", lambda _root: (model, {}))
    monkeypatch.setattr(app, "_sample_observation_window", lambda _paths: _sample_window())
    monkeypatch.setattr(app, "predict_next_day", _fake_prediction)
    monkeypatch.setattr(app, "artifact_signature", lambda *_args, **_kwargs: "fixture-signature")
    monkeypatch.setattr(
        app,
        "_cached_artifacts",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        app,
        "_show_replay",
        lambda _artifacts: app.st.title("Historical replay mode"),
    )
    return AppTest.from_string("import app\napp.main()", default_timeout=30)


def _click_button(at: Any, label: str) -> Any:
    button = next(button for button in at.button if button.label == label)
    button.click()
    return at.run()


def _saved_prediction(at: Any) -> dict[str, Any]:
    return at.session_state[app.PREDICTION_STATE_KEY]


def test_real_streamlit_first_load_edit_stale_rerun_and_guided_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _new_app_test(monkeypatch).run()
    assert at.title[0].value == "SunReserve next-day prediction"
    assert "Edit or confirm the seven rows" in " ".join(at.info.values)
    assert app.PREDICTION_STATE_KEY not in at.session_state

    at = _click_button(at, "Run fresh Ridge prediction")
    original = _saved_prediction(at).copy()
    assert original["prediction"]["target_date"] == "2025-01-01"

    editor_key = f"{app.OBSERVATION_EDITOR_KEY}_0"
    at.session_state[editor_key] = {
        "edited_rows": {6: {app.RADIATION: 8.0}},
        "added_rows": [],
        "deleted_rows": [],
    }
    at.run()
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 8.0
    assert any("Saved prediction is stale" in warning.value for warning in at.warning)
    assert _saved_prediction(at) == original

    at = _click_button(at, "Run fresh Ridge prediction")
    refreshed = _saved_prediction(at)
    assert refreshed["fingerprint"] != original["fingerprint"]
    assert refreshed["observations"][-1][app.RADIATION] == 8.0
    assert not any("Saved prediction is stale" in warning.value for warning in at.warning)

    guided = _new_app_test(monkeypatch).run()
    at = _click_button(guided, "Apply the guided input change")
    assert app.PREDICTION_STATE_KEY not in at.session_state
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 8.0
    assert any("Guided value changed" in info.value for info in at.info)
    at = _click_button(at, "Run fresh Ridge prediction")
    assert app.PREDICTION_STATE_KEY in at.session_state


def test_real_streamlit_assumption_change_marks_stale_until_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _new_app_test(monkeypatch).run()
    at = _click_button(at, "Run fresh Ridge prediction")
    previous = _saved_prediction(at).copy()

    demand = next(item for item in at.number_input if item.label == "Assumed daily demand (kWh)")
    demand.set_value(18.0)
    at.run()
    assert any("Saved prediction is stale" in warning.value for warning in at.warning)
    assert _saved_prediction(at) == previous

    at = _click_button(at, "Run fresh Ridge prediction")
    assert _saved_prediction(at)["assumptions"]["demand"] == 18.0
    assert not any("Saved prediction is stale" in warning.value for warning in at.warning)


def test_real_streamlit_csv_validation_reset_and_same_file_reimport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    at = _new_app_test(monkeypatch).run()
    original_radiation = at.session_state[app.OBSERVATION_EDITOR_DATA_KEY][app.RADIATION].tolist()
    valid_rows = _sample_window()
    valid_rows.loc[6, app.RADIATION] = 9.5
    valid_payload = valid_rows.to_csv(index=False).encode("utf-8")
    invalid_rows = _sample_window()
    invalid_rows.loc[2, "T2M"] = -999
    invalid_payload = invalid_rows.to_csv(index=False).encode("utf-8")

    uploader_key = f"{app.CSV_UPLOAD_KEY}_0"
    at.file_uploader(uploader_key).set_value(("bad.csv", invalid_payload, "text/csv"))
    at.run()
    assert any("NASA POWER -999" in error.value for error in at.error)
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY][app.RADIATION].tolist() == (
        original_radiation
    )

    at.file_uploader(uploader_key).set_value(("observations.csv", valid_payload, "text/csv"))
    at.run()
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 9.5
    assert any("Imported seven observations" in success.value for success in at.success)

    at = _click_button(at, "Reset to historical example")
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY][app.RADIATION].tolist() == (
        original_radiation
    )
    next_uploader_key = f"{app.CSV_UPLOAD_KEY}_{at.session_state[app.CSV_UPLOAD_VERSION_KEY]}"
    at.file_uploader(next_uploader_key).set_value(("observations.csv", valid_payload, "text/csv"))
    at.run()
    assert at.session_state[app.OBSERVATION_EDITOR_DATA_KEY].iloc[-1][app.RADIATION] == 9.5
    assert any("Imported seven observations" in success.value for success in at.success)


def test_real_streamlit_mode_switch_and_session_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _new_app_test(monkeypatch).run()
    second = _new_app_test(monkeypatch).run()
    first = _click_button(first, "Run fresh Ridge prediction")
    saved = _saved_prediction(first).copy()
    assert app.PREDICTION_STATE_KEY not in second.session_state

    first.radio[0].set_value("Historical replay")
    first.run()
    assert "Historical replay mode" in [title.value for title in first.title]

    first.radio[0].set_value("Frozen 2025 results")
    first.run()
    assert "Frozen 2025 results" in [title.value for title in first.title]

    first.radio[0].set_value("Next-day AI prediction")
    first.run()
    assert _saved_prediction(first) == saved
    assert app.PREDICTION_STATE_KEY not in second.session_state
