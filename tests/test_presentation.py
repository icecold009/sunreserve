import hashlib
import io
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path

import app as streamlit_app
import numpy as np
import pandas as pd
import pytest

from sunreserve.config import ExperimentConfig
from sunreserve.data import ProjectPaths, request_url
from sunreserve.decision import (
    APPROACHES,
    BUFFER_FRACTIONS,
    CAPACITY_LEVELS_KWH,
    DEMAND_LEVELS_KWH,
    evaluate_decision_scenarios,
)
from sunreserve.evaluate import (
    ARTIFACT_SCHEMA_VERSION,
    DEVELOPMENT_ROWS_DIGEST_FORMAT,
    PREDICTION_CLIP_RULE,
    _sha256_prepared_csv_prefix,
    forecast_metrics,
)
from sunreserve.presentation import (
    ArtifactError,
    _canonical_hash,
    _current_code_hashes,
    artifact_signature,
    list_validation_experiments,
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
        "selection_metric": config.primary_metric,
        "training_only_low_solar_threshold_kwh_m2_day": 3.5,
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
    development_count = (config.validation_end - config.start_date).days + 1
    prepared_daily_bytes = paths.daily_path.read_bytes()
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "sunreserve_validation_selection",
        "status": "validation_only; final test not loaded, predicted, scored, or reported",
        "configuration": config.to_dict(),
        "configuration_sha256": _canonical_hash(config.to_dict()),
        "code_hashes": _current_code_hashes(code_root),
        "target": "target_radiation_kwh_m2_day",
        "target_units": "kWh/m^2/day",
        "input_hashes": {
            "raw_response_sha256": raw_digest,
            "quality_report_sha256": hashlib.sha256(paths.quality_path.read_bytes()).hexdigest(),
            "prepared_daily_csv_sha256": quality["prepared_daily_csv_sha256"],
            "prepared_development_rows_sha256": _sha256_prepared_csv_prefix(
                prepared_daily_bytes, development_count
            ),
            "prepared_development_rows_digest_format": DEVELOPMENT_ROWS_DIGEST_FORMAT,
        },
        "development_artifact_hashes": development_artifact_hashes,
        "validation_target_period": [
            config.validation_start.isoformat(),
            config.validation_end.isoformat(),
        ],
        "model_selection": {
            "metric": config.primary_metric,
            "selection_partition": "validation",
            "tie_order": ["persistence", "seasonal_mean", "ridge"],
            "selected_model": "ridge",
            "validation_metrics": metrics,
            "prediction_clip_rule": PREDICTION_CLIP_RULE,
            "training_only_low_solar_threshold_kwh_m2_day": 3.5,
            "low_solar_quantile": config.low_solar_quantile,
            "quantile_method": "numpy linear",
            "validation_refit": False,
        },
        "decision_simulation": {
            "partition": "validation",
            "primary_scenario": {
                "panel_area_m2": config.panel_area_m2,
                "efficiency": config.conversion_efficiency,
                "derating": config.derating_factor,
                "demand_kwh": config.daily_demand_kwh,
                "capacity_kwh": config.reserve_cap_kwh,
                "buffer_kwh": config.reserve_buffer_kwh,
            },
            "sensitivity_grid": {
                "demand_kwh": list(DEMAND_LEVELS_KWH),
                "capacity_kwh": list(CAPACITY_LEVELS_KWH),
                "buffer_fraction_of_demand": list(BUFFER_FRACTIONS),
                "scenario_count": 27,
            },
            "approaches": list(APPROACHES),
            "metrics": decision_results.metrics.to_dict(orient="records"),
        },
        "created_utc": "2026-11-14T00:00:00+00:00",
        "code_revision": "fixture",
    }
    (validation_dir / "selection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _replace_bound_output(paths: ProjectPaths, filename: str, payload: bytes) -> None:
    (paths.validation_dir / filename).write_bytes(payload)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["development_artifact_hashes"][filename] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _copy_validation_experiment(root: Path, experiment_id: str) -> Path:
    paths = ProjectPaths(root)
    experiment_dir = root / "artifacts" / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True)
    validation_dir = experiment_dir / "validation"
    shutil.copytree(paths.validation_dir, validation_dir)
    selection_path = validation_dir / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["created_utc"] = "selected-experiment-fixture"
    model_path = experiment_dir / "models" / "ridge_model.json"
    model_path.parent.mkdir(parents=True)
    model_bytes = b"fixture-model"
    model_path.write_bytes(model_bytes)
    selection["model_bundle"] = {
        "path": f"artifacts/experiments/{experiment_id}/models/ridge_model.json",
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    (experiment_dir / "config.json").write_text("{}", encoding="utf-8")
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "sunreserve_validation_experiment",
                "run_id": experiment_id,
                "model_path": f"artifacts/experiments/{experiment_id}/models/ridge_model.json",
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "selection_manifest_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return experiment_dir


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


def test_development_row_digest_uses_persisted_prefix_and_excludes_final_rows(tmp_path) -> None:
    write_demo_fixture(tmp_path)
    config = ExperimentConfig()
    paths = ProjectPaths(tmp_path)
    manifest = json.loads((paths.validation_dir / "selection_manifest.json").read_text())
    payload = paths.daily_path.read_bytes()
    lines = payload.splitlines()
    final_test_row = lines[-1].split(b",")
    final_test_row[2] = b"999.0"
    lines[-1] = b",".join(final_test_row)
    changed_payload = b"\n".join(lines) + (b"\n" if payload.endswith(b"\n") else b"")

    development_count = (config.validation_end - config.start_date).days + 1
    development_row = lines[1].split(b",")
    development_row[2] = b"999.0"
    development_changed_lines = payload.splitlines()
    development_changed_lines[1] = b",".join(development_row)
    development_changed_payload = b"\n".join(development_changed_lines) + (
        b"\n" if payload.endswith(b"\n") else b""
    )

    expected_digest = manifest["input_hashes"]["prepared_development_rows_sha256"]
    assert manifest["input_hashes"]["prepared_development_rows_digest_format"] == (
        DEVELOPMENT_ROWS_DIGEST_FORMAT
    )
    assert _sha256_prepared_csv_prefix(payload, development_count) == expected_digest
    assert _sha256_prepared_csv_prefix(changed_payload, development_count) == expected_digest
    assert (
        _sha256_prepared_csv_prefix(development_changed_payload, development_count)
        != expected_digest
    )


def test_legacy_row_digest_fallback_requires_verified_prepared_csv_bytes(tmp_path) -> None:
    write_demo_fixture(tmp_path)
    paths = ProjectPaths(tmp_path)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hashes = manifest["input_hashes"]
    input_hashes.pop("prepared_development_rows_digest_format")

    config = ExperimentConfig()
    development_count = (config.validation_end - config.start_date).days + 1
    legacy_rows = pd.read_csv(
        io.BytesIO(paths.daily_path.read_bytes()),
        nrows=development_count,
        parse_dates=["date"],
    ).set_index("date")
    legacy_payload = (
        legacy_rows.rename_axis("date").to_csv(index_label="date", lineterminator="\n").encode()
    )
    input_hashes["prepared_development_rows_sha256"] = hashlib.sha256(legacy_payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    load_demo_artifacts(tmp_path)
    paths.daily_path.write_bytes(paths.daily_path.read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match="Prepared daily bytes do not match"):
        load_demo_artifacts(tmp_path)


def test_missing_quality_report_has_actionable_artifact_error(tmp_path) -> None:
    write_demo_fixture(tmp_path)
    paths = ProjectPaths(tmp_path)
    paths.quality_path.unlink()

    with pytest.raises(ArtifactError, match="data quality report is missing"):
        load_demo_artifacts(tmp_path)


def test_experiment_selector_lists_and_loads_scoped_validation_artifacts(tmp_path) -> None:
    write_demo_fixture(tmp_path)
    paths = ProjectPaths(tmp_path)
    shipped_manifest = paths.validation_dir / "selection_manifest.json"
    shipped_before = shipped_manifest.read_bytes()
    _copy_validation_experiment(tmp_path, "trial-1")
    incomplete = tmp_path / "artifacts" / "experiments" / "incomplete"
    incomplete.mkdir()
    (incomplete / "experiment_manifest.json").write_text("{}", encoding="utf-8")

    assert list_validation_experiments(tmp_path) == ["trial-1"]
    selected = load_demo_artifacts(tmp_path, experiment_id="trial-1")
    shipped = load_demo_artifacts(tmp_path)

    assert selected.manifest["created_utc"] == "selected-experiment-fixture"
    assert shipped.manifest["created_utc"] != "selected-experiment-fixture"
    assert shipped_manifest.read_bytes() == shipped_before


def test_experiment_selection_rejects_unsafe_or_unlisted_ids(tmp_path) -> None:
    write_demo_fixture(tmp_path)
    with pytest.raises(ArtifactError, match="safe run identifier"):
        load_demo_artifacts(tmp_path, experiment_id="../escape")
    with pytest.raises(ArtifactError, match="missing"):
        load_demo_artifacts(tmp_path, experiment_id="not-created")


def test_loader_parses_the_exact_checksum_verified_output_snapshot(demo_root, monkeypatch) -> None:
    paths = ProjectPaths(demo_root)
    prediction_path = paths.validation_dir / "validation_predictions.csv"
    original_predictions = prediction_path.read_bytes()
    required_output_paths = {
        paths.validation_dir / name
        for name in (
            "selection_manifest.json",
            "validation_predictions.csv",
            "validation_metrics.json",
            "decision_predictions.csv",
            "decision_metrics.csv",
        )
    }
    read_counts: dict[str, int] = {}
    original_read_bytes = Path.read_bytes

    def capture_then_race(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path in required_output_paths:
            read_counts[path.name] = read_counts.get(path.name, 0) + 1
            if path == prediction_path:
                path.write_bytes(b"this changed after its verified snapshot was captured")
        return payload

    monkeypatch.setattr(Path, "read_bytes", capture_then_race)
    try:
        artifacts = load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert len(artifacts.forecasts) == 365
        assert read_counts == {
            "selection_manifest.json": 1,
            "validation_predictions.csv": 1,
            "decision_predictions.csv": 1,
            "decision_metrics.csv": 1,
            "validation_metrics.json": 1,
        }
    finally:
        prediction_path.write_bytes(original_predictions)


@pytest.mark.parametrize("defect", ["duplicate_key", "nonfinite_number"])
def test_loader_rejects_duplicate_and_nonfinite_json(demo_root, defect) -> None:
    paths = ProjectPaths(demo_root)
    metrics_path = paths.validation_dir / "validation_metrics.json"
    manifest_path = paths.validation_dir / "selection_manifest.json"
    original_metrics = metrics_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    if defect == "duplicate_key":
        payload = b'{"schema_version":2,"schema_version":2}'
    else:
        metrics = json.loads(original_metrics)
        metrics["validation_metrics"]["ridge"]["mae"] = float("inf")
        payload = json.dumps(metrics).encode("utf-8")
    try:
        _replace_bound_output(paths, "validation_metrics.json", payload)
        with pytest.raises(ArtifactError, match="cannot be read") as error:
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert error.value.category == "corrupt"
    finally:
        metrics_path.write_bytes(original_metrics)
        manifest_path.write_bytes(original_manifest)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_forecast_column",
        "unexpected_forecast_column",
        "wrong_units",
        "negative_forecast",
        "nonfinite_decision_value",
        "duplicate_decision_key",
        "unexpected_scenario",
        "wrong_sample_count",
        "wrong_metric_total",
    ],
)
def test_loader_rejects_concrete_schema_and_value_defects(demo_root, defect) -> None:
    paths = ProjectPaths(demo_root)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    original_manifest = manifest_path.read_bytes()
    csv_name = None
    original_csv = None
    try:
        if defect == "wrong_units":
            manifest = json.loads(original_manifest)
            manifest["target_units"] = "MJ/m^2/day"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expected_message = "target or units"
            expected_category = "incompatible"
        elif defect in {
            "missing_forecast_column",
            "unexpected_forecast_column",
            "negative_forecast",
        }:
            csv_name = "validation_predictions.csv"
            csv_path = paths.validation_dir / csv_name
            original_csv = csv_path.read_bytes()
            frame = pd.read_csv(io.BytesIO(original_csv))
            if defect == "missing_forecast_column":
                frame = frame.drop(columns=["ridge_prediction_kwh_m2_day"])
                expected_message = "missing required fields"
                expected_category = "incompatible"
            elif defect == "unexpected_forecast_column":
                frame["unrecognized"] = 1
                expected_message = "unsupported fields"
                expected_category = "incompatible"
            else:
                frame.loc[0, "target_radiation_kwh_m2_day"] = -0.5
                expected_message = "nonnegative"
                expected_category = "incompatible"
            payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            _replace_bound_output(paths, csv_name, payload)
        else:
            csv_name = (
                "validation_predictions.csv"
                if defect == "nonfinite_decision_value"
                else "decision_metrics.csv"
                if defect in {"wrong_sample_count", "wrong_metric_total"}
                else "decision_predictions.csv"
            )
            csv_path = paths.validation_dir / csv_name
            original_csv = csv_path.read_bytes()
            frame = pd.read_csv(io.BytesIO(original_csv))
            if defect == "nonfinite_decision_value":
                csv_name = "decision_predictions.csv"
                csv_path = paths.validation_dir / csv_name
                original_csv = csv_path.read_bytes()
                frame = pd.read_csv(io.BytesIO(original_csv))
                frame.loc[0, "actual_generation_proxy_kwh_day"] = float("inf")
                expected_message = "non-finite"
                expected_category = "corrupt"
            elif defect == "duplicate_decision_key":
                frame.loc[len(frame) - 1, ["scenario_id", "approach", "target_date"]] = frame.loc[
                    0, ["scenario_id", "approach", "target_date"]
                ].to_numpy()
                expected_message = "duplicate scenario/date rows"
                expected_category = "incompatible"
            elif defect == "unexpected_scenario":
                frame.loc[0, "scenario_id"] = "unexpected"
                expected_message = "scenario IDs"
                expected_category = "incompatible"
            elif defect == "wrong_sample_count":
                frame.loc[0, "sample_count"] = 364
                expected_message = "invalid sample count"
                expected_category = "incompatible"
            else:
                frame.loc[0, "total_simulated_shortfall_kwh"] += 1.0
                expected_message = "does not match predictions"
                expected_category = "stale"
            payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            _replace_bound_output(paths, csv_name, payload)

        with pytest.raises(ArtifactError, match=expected_message) as error:
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
        assert error.value.category == expected_category
    finally:
        if csv_name is not None and original_csv is not None:
            (paths.validation_dir / csv_name).write_bytes(original_csv)
        manifest_path.write_bytes(original_manifest)


def test_artifact_signature_hashes_content_when_size_and_mtime_are_unchanged(demo_root) -> None:
    paths = ProjectPaths(demo_root)
    original = paths.raw_path.read_bytes()
    stat = paths.raw_path.stat()
    signature_before = artifact_signature(demo_root, code_root=Path(__file__).resolve().parents[1])
    try:
        paths.raw_path.write_bytes(original.replace(b"offline", b"OFFLINE", 1))
        os.utime(paths.raw_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        changed_stat = paths.raw_path.stat()
        assert changed_stat.st_size == stat.st_size
        assert changed_stat.st_mtime_ns == stat.st_mtime_ns
        signature_after = artifact_signature(
            demo_root, code_root=Path(__file__).resolve().parents[1]
        )
        assert signature_after != signature_before
    finally:
        paths.raw_path.write_bytes(original)
        os.utime(paths.raw_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_artifact_signature_tracks_presentation_loader_source(tmp_path) -> None:
    code_root = tmp_path / "code"
    presentation_source = code_root / "src" / "sunreserve" / "presentation.py"
    presentation_source.parent.mkdir(parents=True)
    presentation_source.write_text("version one", encoding="utf-8")

    signature_before = artifact_signature(tmp_path / "demo", code_root=code_root)
    assert (str(presentation_source), "sha256:" + hashlib.sha256(b"version one").hexdigest()) in (
        signature_before
    )

    presentation_source.write_text("version two", encoding="utf-8")
    signature_after = artifact_signature(tmp_path / "demo", code_root=code_root)
    assert signature_after != signature_before


@pytest.mark.parametrize(
    "filename",
    [
        "selection_manifest.json",
        "validation_predictions.csv",
        "validation_metrics.json",
        "decision_predictions.csv",
        "decision_metrics.csv",
    ],
)
def test_removing_required_output_changes_signature_and_keeps_actionable_error(
    demo_root, filename
) -> None:
    paths = ProjectPaths(demo_root)
    target = paths.validation_dir / filename
    original = target.read_bytes()
    code_root = Path(__file__).resolve().parents[1]
    signature_before = artifact_signature(demo_root, code_root=code_root)
    try:
        target.unlink()
        missing_signature = artifact_signature(demo_root, code_root=code_root)
        assert missing_signature != signature_before
        assert missing_signature == artifact_signature(demo_root, code_root=code_root)
        with pytest.raises(ArtifactError, match="Missing cached validation output") as missing:
            load_demo_artifacts(demo_root, code_root=code_root)
        assert missing.value.category == "missing"
        assert "Restore the shipped artifacts or create and select a validation experiment" in str(
            missing.value
        )
    finally:
        target.write_bytes(original)


def test_stored_artifacts_load_with_changed_viewer_sources(demo_root, tmp_path) -> None:
    code_root = tmp_path / "changed-source"
    source_dir = code_root / "src" / "sunreserve"
    source_dir.mkdir(parents=True)
    for name in (
        "config.py",
        "data.py",
        "features.py",
        "train.py",
        "evaluate.py",
        "decision.py",
        "__main__.py",
        "inference.py",
    ):
        (source_dir / name).write_text(f"changed {name}", encoding="utf-8")
    artifacts = load_demo_artifacts(demo_root, code_root=code_root)
    assert artifacts.manifest["code_hashes"]
    assert artifacts.manifest["code_revision"] == "fixture"


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"code_hashes": {}, "code_revision": "fixture"},
        {"code_hashes": {"x": "bad"}, "code_revision": 2},
    ],
)
def test_loader_rejects_malformed_producer_identity(demo_root, identity) -> None:
    path = ProjectPaths(demo_root).validation_dir / "selection_manifest.json"
    original = path.read_bytes()
    try:
        manifest = json.loads(original)
        manifest.pop("code_hashes", None)
        manifest.pop("code_revision", None)
        manifest.update(identity)
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactError, match="producer identity"):
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
    finally:
        path.write_bytes(original)


def test_loader_rejects_tampered_checksum_bound_output(demo_root) -> None:
    path = ProjectPaths(demo_root).validation_dir / "validation_predictions.csv"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ArtifactError, match="checksum"):
            load_demo_artifacts(demo_root, code_root=Path(__file__).resolve().parents[1])
    finally:
        path.write_bytes(original)


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
    assert any(label == "Observed NASA POWER radiation (target)" for label, _ in fake_ui.metrics)
    assert any(label == "Simulated predicted electricity proxy" for label, _ in fake_ui.metrics)
    assert any(label == "Post-hoc simulated shortfall" for label, _ in fake_ui.metrics)
    assert any(label == "Simulated realized electricity proxy" for label, _ in fake_ui.metrics)
    assert any("@media (max-width: 640px)" in str(item) for item in fake_ui.writes)
    prepared_value = next(
        value
        for label, value in fake_ui.metrics
        if label == "Prepared reserve (forecast, after capacity cap)"
    )
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
    assert (
        "git restore --source=HEAD -- artifacts/validation models/ridge_model.json"
        in (fake_ui.code_blocks[0])
    )
    assert "python -m sunreserve validate --run-id recovery-2026-11-19" in (fake_ui.code_blocks[1])
    assert any("never runs those commands" in caption for caption in fake_ui.captions)
