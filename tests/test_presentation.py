import hashlib
import json
from datetime import timedelta
from pathlib import Path

import app as streamlit_app
import numpy as np
import pandas as pd
import pytest

from sunreserve.config import ExperimentConfig
from sunreserve.data import ProjectPaths, request_url
from sunreserve.decision import evaluate_decision_scenarios
from sunreserve.evaluate import ARTIFACT_SCHEMA_VERSION, forecast_metrics
from sunreserve.presentation import (
    ArtifactError,
    _canonical_hash,
    _current_code_hashes,
    _development_data_hash,
    load_demo_artifacts,
)


def write_demo_fixture(root: Path) -> None:
    config = ExperimentConfig()
    code_root = Path(__file__).resolve().parents[1]
    paths = ProjectPaths(root)
    paths.daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily_dates = pd.date_range(config.start_date, config.end_date, freq="D")
    offset = np.arange(len(daily_dates), dtype=float)
    daily = pd.DataFrame(
        {
            "date": daily_dates,
            "ALLSKY_SFC_SW_DWN": 4.0 + 0.5 * np.sin(offset / 15),
            "T2M": 23.0 + 2 * np.sin(offset / 30),
            "RH2M": 55.0 + 5 * np.cos(offset / 40),
        }
    )
    daily.to_csv(paths.daily_path, index=False, lineterminator="\n")
    raw_payload = b"offline-demo-fixture-source"
    raw_digest = hashlib.sha256(raw_payload).hexdigest()
    paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_path.write_bytes(raw_payload)
    paths.sidecar_path.write_text(
        json.dumps({"request_url": request_url(config), "sha256": raw_digest}),
        encoding="utf-8",
    )
    quality = {
        "schema_version": 1,
        "source": config.source,
        "requested_start": config.start_date.isoformat(),
        "requested_end": config.end_date.isoformat(),
        "requested_location": {"latitude": config.latitude, "longitude": config.longitude},
        "time_standard": config.time_standard,
        "units": {
            "ALLSKY_SFC_SW_DWN": "kW-hr/m^2/day",
            "T2M": "C",
            "RH2M": "%",
        },
        "calendar_days": len(daily),
        "complete_case_days": len(daily),
        "excluded_days": 0,
        "missing_counts_by_variable": {key: 0 for key in config.parameters},
        "raw_sha256": raw_digest,
        "prepared_daily_csv_sha256": hashlib.sha256(paths.daily_path.read_bytes()).hexdigest(),
    }
    paths.quality_path.parent.mkdir(parents=True, exist_ok=True)
    paths.quality_path.write_text(json.dumps(quality), encoding="utf-8")

    validation_dates = pd.date_range(config.validation_start, config.validation_end, freq="D")
    phase = np.arange(len(validation_dates), dtype=float)
    actual = 4.0 + 0.7 * np.sin(phase / 19)
    forecasts = pd.DataFrame(
        {
            "issue_date": validation_dates - timedelta(days=1),
            "target_date": validation_dates,
            "target_radiation_kwh_m2_day": actual,
            "persistence_prediction_kwh_m2_day": actual + 0.15 * np.cos(phase / 8),
            "seasonal_mean_prediction_kwh_m2_day": actual + 0.3 * np.sin(phase / 11),
            "ridge_prediction_kwh_m2_day": actual + 0.08 * np.cos(phase / 6),
        }
    )
    validation_dir = paths.validation_dir
    validation_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(
        validation_dir / "validation_predictions.csv", index=False, lineterminator="\n"
    )
    metrics = {
        name: forecast_metrics(
            actual,
            forecasts[column],
            forecasts["target_date"],
            low_solar_threshold=3.5,
        )
        for name, column in {
            "persistence": "persistence_prediction_kwh_m2_day",
            "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
            "ridge": "ridge_prediction_kwh_m2_day",
        }.items()
    }
    decision_results = evaluate_decision_scenarios(forecasts, config, selected_model="ridge")
    decision_results.predictions.to_csv(
        validation_dir / "decision_predictions.csv", index=False, lineterminator="\n"
    )
    decision_results.metrics.to_csv(
        validation_dir / "decision_metrics.csv", index=False, lineterminator="\n"
    )
    validation_metrics_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "selected_model": "ridge",
        "validation_metrics": metrics,
    }
    (validation_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics_payload), encoding="utf-8"
    )
    development_artifact_hashes = {
        name: hashlib.sha256((validation_dir / name).read_bytes()).hexdigest()
        for name in (
            "validation_predictions.csv",
            "validation_metrics.json",
            "decision_predictions.csv",
            "decision_metrics.csv",
        )
    }
    daily_development = daily.loc[daily["date"] <= pd.Timestamp(config.validation_end)].set_index(
        "date"
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "sunreserve_validation_selection",
        "status": "validation_only; final test not loaded, predicted, scored, or reported",
        "configuration": config.to_dict(),
        "configuration_sha256": _canonical_hash(config.to_dict()),
        "code_hashes": _current_code_hashes(code_root),
        "input_hashes": {
            "raw_response_sha256": raw_digest,
            "quality_report_sha256": hashlib.sha256(paths.quality_path.read_bytes()).hexdigest(),
            "prepared_daily_csv_sha256": quality["prepared_daily_csv_sha256"],
            "prepared_development_rows_sha256": _development_data_hash(daily_development),
        },
        "development_artifact_hashes": development_artifact_hashes,
        "validation_target_period": [
            config.validation_start.isoformat(),
            config.validation_end.isoformat(),
        ],
        "model_selection": {
            "selected_model": "ridge",
            "validation_metrics": metrics,
        },
        "decision_simulation": {
            "sensitivity_grid": {"scenario_count": 27},
        },
        "created_utc": "2026-11-14T00:00:00+00:00",
        "code_revision": "fixture",
    }
    (validation_dir / "selection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("sunreserve-demo")
    write_demo_fixture(root)
    return root


def test_offline_loader_accepts_complete_validation_fixture(demo_root) -> None:
    artifacts = load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])

    assert len(artifacts.forecasts) == 365
    assert len(artifacts.dates) == 365
    assert artifacts.dates[0].isoformat() == "2022-01-01"
    assert artifacts.dates[-1].isoformat() == "2022-12-31"
    assert artifacts.manifest["model_selection"]["selected_model"] == "ridge"
    assert artifacts.decisions["scenario_id"].nunique() == 28
    assert set(artifacts.decisions["approach"]) == {
        "persistence",
        "seasonal_mean",
        "ridge",
        "always_prepare_backup",
    }


def test_loader_explains_missing_corrupt_incompatible_and_stale_artifacts(demo_root, tmp_path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(ArtifactError, match="Missing cached validation output") as missing:
        load_demo_artifacts(missing_root, code_root=Path(__file__).resolve().parents[1])
    assert missing.value.category == "missing"

    manifest_path = ProjectPaths(demo_root).validation_dir / "selection_manifest.json"
    original = manifest_path.read_bytes()
    try:
        manifest_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(ArtifactError, match="cannot be read") as corrupt:
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert corrupt.value.category == "corrupt"

        incompatible_manifest = json.loads(original)
        incompatible_manifest["schema_version"] = 999
        manifest_path.write_text(json.dumps(incompatible_manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match="schema or artifact type") as incompatible:
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert incompatible.value.category == "incompatible"

        manifest_path.write_bytes(original)
        paths = ProjectPaths(demo_root)
        sidecar = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
        sidecar["sha256"] = "stale-source-hash"
        paths.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        with pytest.raises(ArtifactError, match="checksum") as stale:
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert stale.value.category == "stale"
    finally:
        manifest_path.write_bytes(original)
        sidecar_path = ProjectPaths(demo_root).sidecar_path
        sidecar_path.write_text(
            json.dumps(
                {
                    "request_url": request_url(ExperimentConfig()),
                    "sha256": hashlib.sha256(
                        ProjectPaths(demo_root).raw_path.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )


class RecordingStreamlit:
    """Small offline UI double for checking content sent to Streamlit widgets."""

    def __init__(self, *, values=None):
        self.values = values or {}
        self.headings = []
        self.warnings = []
        self.captions = []
        self.metrics = []
        self.charts = []
        self.tables = []
        self.infos = []
        self.writes = []
        self.code_blocks = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def title(self, value):
        self.headings.append(value)

    def header(self, value):
        self.headings.append(value)

    def subheader(self, value):
        self.headings.append(value)

    def warning(self, value):
        self.warnings.append(value)

    def caption(self, value):
        self.captions.append(value)

    def selectbox(self, label, options, index=0, **_kwargs):
        return self.values.get(label, options[index])

    def date_input(self, label, value, **_kwargs):
        return self.values.get(label, value)

    def columns(self, count):
        return [self] * count

    def metric(self, label, value, **_kwargs):
        self.metrics.append((label, value))

    def line_chart(self, data, **_kwargs):
        self.charts.append(data)

    def dataframe(self, data, **_kwargs):
        self.tables.append(data.copy())

    def info(self, value):
        self.infos.append(value)

    def error(self, value):
        self.warnings.append(value)

    def write(self, value):
        self.writes.append(value)

    def markdown(self, value, **_kwargs):
        self.writes.append(value)

    def code(self, value, **_kwargs):
        self.code_blocks.append(value)

    def expander(self, _label):
        return self

    def json(self, value, **_kwargs):
        self.writes.append(value)


def test_streamlit_render_shows_selected_date_scenario_chart_and_failures(demo_root, monkeypatch):
    artifacts = load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
    fake_ui = RecordingStreamlit(
        values={
            "Forecast approach": "persistence",
            "Saved reserve scenario": "grid_d20_c10_b0",
            "Validation target date": pd.Timestamp("2022-06-01").date(),
        }
    )
    monkeypatch.setattr(streamlit_app, "st", fake_ui)

    streamlit_app._show_replay(artifacts)

    assert "SunReserve historical replay" in fake_ui.headings
    assert any("2022 validation replay" in message for message in fake_ui.warnings)
    assert any(label == "NASA POWER solar radiation" for label, _ in fake_ui.metrics)
    assert any(label == "Simulated predicted electricity proxy" for label, _ in fake_ui.metrics)
    assert any(label == "Simulated shortfall" for label, _ in fake_ui.metrics)
    assert any("@media (max-width: 640px)" in str(item) for item in fake_ui.writes)
    prepared_value = next(value for label, value in fake_ui.metrics if label == "Prepared reserve")
    assert prepared_value.endswith(" kWh/day")
    assert len(prepared_value) <= streamlit_app.MOBILE_METRIC_VALUE_MAX_CHARS
    assert any("simulated kWh/day quantities" in caption for caption in fake_ui.captions)
    assert len(fake_ui.charts) == 1
    assert list(fake_ui.charts[0].columns) == [
        "NASA POWER radiation",
        "Persistence baseline prediction",
    ]
    assert len(fake_ui.tables) == 2
    assert set(fake_ui.tables[0]["Approach"]) == {
        "Persistence baseline",
        "Training-only seasonal mean baseline",
        "Ridge regression",
        "Always prepare backup reference",
    }
    assert set(fake_ui.tables[1]["example_type"]) == {
        "Largest overprediction",
        "Largest underprediction",
    }
    assert any("Largest forecast errors" in heading for heading in fake_ui.headings)
    assert any("historical daily radiation" in str(item) for item in fake_ui.writes)


@pytest.mark.parametrize("value", [0.0, 15.0, 20.0])
def test_energy_metric_format_fits_mobile_character_budget(value):
    rendered = streamlit_app._format_kwh_day(value)

    assert rendered.endswith(" kWh/day")
    assert len(rendered) <= streamlit_app.MOBILE_METRIC_VALUE_MAX_CHARS


def test_missing_artifact_view_gives_offline_recovery_instructions(monkeypatch):
    fake_ui = RecordingStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_ui)

    streamlit_app._show_load_failure(ArtifactError("missing", "No cached reports."))

    assert any("missing" in warning for warning in fake_ui.warnings)
    assert "python -m sunreserve validate" in fake_ui.code_blocks[0]
    assert any("never runs those commands" in caption for caption in fake_ui.captions)
