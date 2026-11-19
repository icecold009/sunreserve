from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sunreserve import evaluate as evaluate_module
from sunreserve.config import ExperimentConfig
from sunreserve.data import DataError, ProjectPaths, fetch_raw, prepare_daily
from sunreserve.evaluate import (
    FINAL_RUN_SCHEMA_VERSION,
    _sha256_file,
    finalize_run,
    load_finalized_run,
    render_final_report,
    run_experiment_validation,
    run_validation,
)
from sunreserve.inference import (
    load_model_bundle,
    predict_ridge_bundle,
    predict_seasonal_mean_bundle,
)


def _synthetic_power_payload(config: ExperimentConfig) -> bytes:
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    positions = np.arange(len(dates), dtype=float)
    date_keys = [stamp.strftime("%Y%m%d") for stamp in dates]
    radiation = 4.0 + 0.8 * np.sin(positions / 18) + 0.15 * np.cos(positions / 3)
    temperature = 22.0 + 3.0 * np.sin(positions / 365 * 2 * np.pi)
    humidity = 58.0 + 12.0 * np.cos(positions / 365 * 2 * np.pi)
    if len(dates) != 3288:
        raise AssertionError("synthetic integration fixture should cover 2016–2024")
    document: dict[str, Any] = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [77.595, 12.972, 840.0]},
        "header": {
            "start": config.start_date.strftime("%Y%m%d"),
            "end": config.end_date.strftime("%Y%m%d"),
            "time_standard": config.time_standard,
        },
        "parameters": {
            "ALLSKY_SFC_SW_DWN": {"units": "kW-hr/m^2/day"},
            "T2M": {"units": "C"},
            "RH2M": {"units": "%"},
        },
        "properties": {
            "parameter": {
                "ALLSKY_SFC_SW_DWN": dict(zip(date_keys, radiation.tolist(), strict=True)),
                "T2M": dict(zip(date_keys, temperature.tolist(), strict=True)),
                "RH2M": dict(zip(date_keys, humidity.tolist(), strict=True)),
            }
        },
    }
    return json.dumps(document, separators=(",", ":")).encode("utf-8")


def _build_frozen_fixture(root: Path) -> tuple[ProjectPaths, str]:
    config = ExperimentConfig()
    paths = ProjectPaths(root)
    payload = _synthetic_power_payload(config)
    fetch_raw(config, paths, downloader=lambda _url: payload)
    prepare_daily(config, paths)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "assumptions.md").write_text(
        "Synthetic integration fixture; assumptions remain hypothetical.\n", encoding="utf-8"
    )
    run_validation(paths, config)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    return paths, _sha256_file(manifest_path)


def _build_finalized_run(root: Path, run_id: str = "loader-fixture") -> Path:
    paths, frozen_digest = _build_frozen_fixture(root)
    return finalize_run(paths, manifest_sha256=frozen_digest, run_id=run_id).run_dir


def _copy_finalized_run(source: Path, destination_root: Path) -> Path:
    destination = destination_root / source.name
    shutil.copytree(source, destination)
    return destination


def _copy_project_root(source_root: Path, destination_root: Path) -> ProjectPaths:
    shutil.copytree(source_root, destination_root, dirs_exist_ok=True)
    return ProjectPaths(destination_root)


def _rewrite_finalized_artifact(run_dir: Path, name: str, payload: bytes) -> None:
    (run_dir / name).write_bytes(payload)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"][name] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture(scope="module")
def finalized_run_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_finalized_run(tmp_path_factory.mktemp("finalized-loader-source"))


@pytest.fixture(scope="module")
def experiment_validation_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("experiment-model-source")
    paths, _ = _build_frozen_fixture(root)
    run_experiment_validation(paths, ExperimentConfig(), "loader-exp")
    return root


def _frozen_run_hashes() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    frozen_runs = ("frozen-test-2023-24", "frozen-test-2025")
    return {
        path.relative_to(repo_root).as_posix(): _sha256_file(path)
        for run_id in frozen_runs
        for path in sorted((repo_root / "artifacts" / "runs" / run_id).rglob("*"))
        if path.is_file()
    }


def _bundle_test_predictions(
    paths: ProjectPaths, model_path: Path, selection: dict[str, Any]
) -> tuple[Any, pd.DataFrame, pd.Series]:
    config = ExperimentConfig.from_dict(selection["configuration"])
    bundle = load_model_bundle(
        model_path,
        expected_sha256=selection["model_bundle"]["sha256"],
    )
    daily = evaluate_module._load_full_daily(paths, config)
    rows, _ = evaluate_module.build_supervised_rows(daily, config)
    test_rows = evaluate_module.split_supervised_rows(rows)["test"]
    return bundle, test_rows, predict_ridge_bundle(bundle, test_rows)


def test_load_finalized_run_accepts_complete_tmp_path_fixture(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)

    loaded = load_finalized_run(run_dir)

    assert loaded["run_manifest"]["run_id"] == run_dir.name
    assert len(loaded["predictions"]) == 731
    assert set(loaded["forecast_metrics"]["metrics"]) == {"persistence", "seasonal_mean", "ridge"}
    assert len(loaded["decision_metrics"]) == 28 * 4
    assert len(loaded["decision_predictions"]) == 731 * 28 * 4
    assert "Synthetic integration fixture" in loaded["assumptions"]


def test_load_finalized_run_hashes_and_parses_one_captured_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finalized_run_source: Path,
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    prediction_path = run_dir / "predictions.csv"
    original_prediction_bytes = prediction_path.read_bytes()
    original_read_bytes = Path.read_bytes
    read_counts: dict[str, int] = {}
    required_names = {
        "run_manifest.json",
        "predictions.csv",
        "forecast_metrics.json",
        "decision_metrics.csv",
        "decision_predictions.csv",
        "data_quality.json",
        "selection_manifest.json",
        "assumptions.md",
    }

    def snapshot_then_mutate(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path.parent == run_dir and path.name in required_names:
            read_counts[path.name] = read_counts.get(path.name, 0) + 1
        if path == prediction_path:
            prediction_path.write_bytes(b"changed after captured snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", snapshot_then_mutate)
    try:
        loaded = load_finalized_run(run_dir)
        assert len(loaded["predictions"]) == 731
        assert read_counts == {name: 1 for name in required_names}
    finally:
        monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
        prediction_path.write_bytes(original_prediction_bytes)


def test_load_finalized_run_rejects_duplicate_json_keys(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    _rewrite_finalized_artifact(
        run_dir,
        "forecast_metrics.json",
        b'{"schema_version":1,"schema_version":1}',
    )

    with pytest.raises(DataError, match="duplicate JSON key"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_malformed_date(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    frame = pd.read_csv(run_dir / "predictions.csv")
    frame.loc[0, "target_date"] = "2024-13-01"
    _rewrite_finalized_artifact(
        run_dir,
        "predictions.csv",
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )

    with pytest.raises(DataError, match="contains an invalid date"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_interior_gap_without_exclusion_record(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    predictions_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path).drop(index=100).reset_index(drop=True)
    prediction_bytes = predictions.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _rewrite_finalized_artifact(run_dir, "predictions.csv", prediction_bytes)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_counts"]["test"] = len(predictions)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="calendar does not reconcile to exclusions"):
        load_finalized_run(run_dir)


def test_load_finalized_run_accepts_legacy_historical_manifest_from_tmp_copy(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    committed_run = repo_root / "artifacts" / "runs" / "frozen-test-2023-24"
    run_dir = tmp_path / committed_run.name
    shutil.copytree(committed_run, run_dir)

    loaded = load_finalized_run(run_dir)

    target_dates = pd.DatetimeIndex(loaded["predictions"]["target_date"])
    assert len(target_dates) == 731
    assert target_dates.equals(pd.date_range("2023-01-01", "2024-12-31", freq="D"))
    assert loaded["run_manifest"]["row_counts"]["test"] == len(target_dates)
    assert loaded["run_manifest"]["row_counts"]["test_excluded"] == 6
    assert "excluded_rows_by_split" not in loaded["run_manifest"]
    assert "excluded_dates_by_split" not in loaded["run_manifest"]


def test_load_finalized_run_rejects_partial_exclusion_metadata(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("excluded_dates_by_split")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="excluded-row accounting is incomplete"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_new_manifest_with_both_exclusion_maps_removed(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("excluded_rows_by_split")
    manifest.pop("excluded_dates_by_split")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="excluded-row accounting is incomplete"):
        load_finalized_run(run_dir)


def test_load_finalized_run_requires_bundle_identity_for_new_runs(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("model_bundle")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="missing its frozen model bundle identity"):
        load_finalized_run(run_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_id", "unknown.model"), ("sha256", "bad-digest"), ("training_rows", True)],
)
def test_load_finalized_run_rejects_invalid_bundle_identity_fields(
    tmp_path: Path, finalized_run_source: Path, field: str, value: object
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_bundle"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="invalid frozen model bundle identity"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_bundle_identity_mismatch_with_selection(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_bundle"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataError, match="does not match the frozen selection manifest"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_extra_columns(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    frame = pd.read_csv(run_dir / "decision_metrics.csv")
    frame["unexpected"] = 1
    _rewrite_finalized_artifact(
        run_dir,
        "decision_metrics.csv",
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )

    with pytest.raises(DataError, match="columns do not match"):
        load_finalized_run(run_dir)


def test_load_finalized_run_rejects_invalid_numeric_output(
    tmp_path: Path, finalized_run_source: Path
) -> None:
    run_dir = _copy_finalized_run(finalized_run_source, tmp_path)
    frame = pd.read_csv(run_dir / "predictions.csv")
    frame.loc[0, "actual_radiation_kwh_m2_day"] = -1.0
    _rewrite_finalized_artifact(
        run_dir,
        "predictions.csv",
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    )

    with pytest.raises(DataError, match="must be nonnegative"):
        load_finalized_run(run_dir)


def test_finalize_reuses_frozen_numeric_bundle_without_refitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finalized_run_source: Path,
) -> None:
    source_root = finalized_run_source.parents[2]
    paths = _copy_project_root(source_root, tmp_path)
    selection_path = paths.validation_dir / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bundle, test_rows, expected = _bundle_test_predictions(paths, paths.model_path, selection)
    expected_persistence = np.maximum(
        test_rows["radiation_today_kwh_m2_day"].to_numpy(dtype=float), 0.0
    )
    expected_seasonal = predict_seasonal_mean_bundle(bundle, test_rows).to_numpy()

    def forbidden_refit(*_args, **_kwargs):
        pytest.fail("finalization must reuse the frozen numeric model bundle")

    monkeypatch.setattr(evaluate_module, "fit_ridge", forbidden_refit)
    monkeypatch.setattr(evaluate_module, "baseline_predictions", forbidden_refit)
    finalized = finalize_run(
        paths,
        manifest_sha256=_sha256_file(selection_path),
        run_id="bundle-reuse-main",
    )
    prediction_frame = pd.read_csv(finalized.run_dir / "predictions.csv")
    manifest = json.loads((finalized.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_identity = {
        "path": selection["model_bundle"]["path"],
        "model_id": bundle.model_id,
        "sha256": bundle.artifact_sha256,
        "training_split": "train",
        "training_rows": bundle.training_rows,
        "training_target_period": [
            bundle.training_start.isoformat(),
            bundle.training_end.isoformat(),
        ],
    }

    assert manifest["model_bundle"] == expected_identity
    assert prediction_frame["ridge_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected.to_numpy(), abs=1e-10
    )
    assert prediction_frame["persistence_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected_persistence, abs=1e-10
    )
    assert prediction_frame["seasonal_mean_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected_seasonal, abs=1e-10
    )


@pytest.mark.parametrize(
    ("source", "failure"),
    [
        (source, failure)
        for source in ("main", "experiment")
        for failure in ("missing", "changed", "invalid")
    ],
)
def test_finalize_model_failures_precede_test_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finalized_run_source: Path,
    experiment_validation_source: Path,
    source: str,
    failure: str,
) -> None:
    source_root = (
        finalized_run_source.parents[2] if source == "main" else experiment_validation_source
    )
    paths = _copy_project_root(source_root, tmp_path)
    experiment_id = "loader-exp" if source == "experiment" else None
    if experiment_id is None:
        selection_path = paths.validation_dir / "selection_manifest.json"
        model_path = paths.model_path
        experiment_manifest_path = None
    else:
        experiment_dir = paths.root / "artifacts" / "experiments" / experiment_id
        selection_path = experiment_dir / "validation" / "selection_manifest.json"
        model_path = experiment_dir / "models" / "ridge_model.json"
        experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    if failure == "missing":
        model_path.unlink()
    elif failure == "changed":
        model_path.write_bytes(model_path.read_bytes() + b"\nchanged")
    else:
        model_payload = json.loads(model_path.read_text(encoding="utf-8"))
        model_payload["ridge"]["coefficients"] = []
        model_bytes = json.dumps(model_payload).encode("utf-8")
        model_path.write_bytes(model_bytes)
        new_model_digest = hashlib.sha256(model_bytes).hexdigest()
        selection["model_bundle"]["sha256"] = new_model_digest
        selection_path.write_text(json.dumps(selection), encoding="utf-8")

    selection_digest = _sha256_file(selection_path)
    if experiment_manifest_path is not None and failure == "invalid":
        experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
        experiment_manifest["model_sha256"] = new_model_digest
        experiment_manifest["selection_manifest_sha256"] = selection_digest
        experiment_manifest_path.write_text(json.dumps(experiment_manifest), encoding="utf-8")

    def forbidden_test_parse(*_args, **_kwargs):
        pytest.fail("invalid frozen bundle must be rejected before test rows are parsed")

    monkeypatch.setattr(evaluate_module, "_load_full_daily", forbidden_test_parse)
    expected_error = "frozen model bundle is invalid" if failure == "invalid" else "model"
    with pytest.raises(DataError, match=expected_error):
        finalize_run(
            paths,
            manifest_sha256=selection_digest,
            run_id=f"reject-{source}-{failure}",
            experiment_id=experiment_id,
        )


def test_offline_prepare_validate_finalize_report_contract(tmp_path, monkeypatch) -> None:
    frozen_hashes = _frozen_run_hashes()
    paths, frozen_digest = _build_frozen_fixture(tmp_path)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert selection["eligible_rows"] == {"train": 2185, "validation": 365}
    assert selection["code_revision"] == "unavailable" or len(selection["code_revision"]) in (
        40,
        64,
    )
    assert selection["working_tree_dirty"] in (True, False, None)
    assert "source_provenance" in selection["code_hashes"]
    assert set(selection["development_artifact_hashes"]) == {
        "validation_predictions.csv",
        "validation_metrics.json",
        "decision_predictions.csv",
        "decision_metrics.csv",
    }
    development_predictions = pd.read_csv(
        paths.validation_dir / "validation_predictions.csv", parse_dates=["target_date"]
    )
    assert development_predictions["target_date"].dt.year.unique().tolist() == [2022]

    def forbidden_test_load(*_args, **_kwargs):
        pytest.fail("frozen-manifest rejection must occur before parsing test-period rows")

    load_full_daily = evaluate_module._load_full_daily
    monkeypatch.setattr(evaluate_module, "_load_full_daily", forbidden_test_load)
    with pytest.raises(DataError, match="SHA-256 does not match"):
        finalize_run(paths, manifest_sha256="0" * 64, run_id="wrong-freeze")
    with pytest.raises(DataError, match="active configuration"):
        finalize_run(
            paths,
            manifest_sha256=frozen_digest,
            run_id="wrong-config",
            expected_config=ExperimentConfig(reserve_buffer_kwh=2.0),
        )

    monkeypatch.setattr(evaluate_module, "_load_full_daily", load_full_daily)
    original_daily = paths.daily_path.read_bytes()
    paths.daily_path.write_bytes(original_daily + b"\n")
    assert _sha256_file(paths.daily_path) != selection["input_hashes"]["prepared_daily_csv_sha256"]
    for artifact_name, expected_hash in selection["development_artifact_hashes"].items():
        assert _sha256_file(paths.validation_dir / artifact_name) == expected_hash, artifact_name
    monkeypatch.setattr(evaluate_module, "_load_full_daily", forbidden_test_load)
    with pytest.raises(DataError, match="prepared daily file hash"):
        finalize_run(paths, manifest_sha256=frozen_digest, run_id="wrong-prepared-hash")
    paths.daily_path.write_bytes(original_daily)
    monkeypatch.setattr(evaluate_module, "_load_full_daily", load_full_daily)

    finalized = finalize_run(paths, manifest_sha256=frozen_digest, run_id="fixture-run")
    assert finalized.row_counts == {
        "train": 2185,
        "validation": 365,
        "test": 731,
    }
    run_dir = finalized.run_dir
    required = {
        "predictions.csv",
        "forecast_metrics.json",
        "decision_metrics.csv",
        "decision_predictions.csv",
        "data_quality.json",
        "selection_manifest.json",
        "run_manifest.json",
        "assumptions.md",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["schema_version"] == FINAL_RUN_SCHEMA_VERSION
    assert run_manifest["test_inspected"] is True
    assert run_manifest["code_revision"] == "unavailable" or len(run_manifest["code_revision"]) in (
        40,
        64,
    )
    assert run_manifest["working_tree_dirty"] in (True, False, None)
    assert "source_provenance" in run_manifest["current_code_hashes"]
    assert run_manifest["public_demo_bundle"]["status"] == "not_created"
    assert run_manifest["row_counts"]["test"] == 731
    assert "test_excluded" not in run_manifest["row_counts"]
    assert run_manifest["excluded_rows_by_split"] == {"train": 6, "validation": 0, "test": 0}
    assert [
        record["target_date"] for record in run_manifest["excluded_dates_by_split"]["train"]
    ] == [f"2016-01-{day:02d}" for day in range(2, 8)]
    assert run_manifest["excluded_dates_by_split"]["validation"] == []
    assert run_manifest["excluded_dates_by_split"]["test"] == []
    assert all(
        set(record) == {"issue_date", "target_date", "missing_or_nonfinite_fields"}
        and record["missing_or_nonfinite_fields"]
        for record in run_manifest["excluded_dates_by_split"]["train"]
    )

    predictions = pd.read_csv(run_dir / "predictions.csv", parse_dates=["target_date"])
    assert set(predictions["split"]) == {"test"}
    assert predictions["target_date"].min() == pd.Timestamp("2023-01-01")
    assert predictions["target_date"].max() == pd.Timestamp("2024-12-31")
    assert {
        "persistence_prediction_kwh_m2_day",
        "seasonal_mean_prediction_kwh_m2_day",
        "ridge_prediction_kwh_m2_day",
    } <= set(predictions.columns)
    forecast_metrics = json.loads((run_dir / "forecast_metrics.json").read_text(encoding="utf-8"))
    assert forecast_metrics["partition"] == "test"
    assert set(forecast_metrics["metrics"]) == {"persistence", "seasonal_mean", "ridge"}
    decisions = pd.read_csv(run_dir / "decision_metrics.csv")
    assert len(decisions) == 28 * 4
    assert set(decisions["partition"]) == {"test"}
    assert {"demand_kwh", "capacity_kwh", "buffer_kwh"} <= set(decisions.columns)
    assert set(decisions["approach"]) == {
        "persistence",
        "seasonal_mean",
        "ridge",
        "always_prepare_backup",
    }
    assert run_manifest["artifact_sha256"]["predictions.csv"] == _sha256_file(
        run_dir / "predictions.csv"
    )

    second_run = finalize_run(paths, manifest_sha256=frozen_digest, run_id="fixture-run-repeat")
    assert (run_dir / "predictions.csv").read_bytes() == (
        second_run.run_dir / "predictions.csv"
    ).read_bytes()
    assert (run_dir / "decision_metrics.csv").read_bytes() == (
        second_run.run_dir / "decision_metrics.csv"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        finalize_run(paths, manifest_sha256=frozen_digest, run_id="fixture-run")
    with pytest.raises(ValueError, match="run_id"):
        finalize_run(paths, manifest_sha256=frozen_digest, run_id="../escape")

    from sunreserve.__main__ import main

    assert main(["--root", str(paths.root), "report", "--run-id", "../outside"]) == 2

    report_path = render_final_report(run_dir)
    report = report_path.read_text(encoding="utf-8")
    assert "Forecast comparison" in report
    assert "Seasonal behavior" in report
    assert "Unfavorable forecast examples" in report
    assert "Fixed sensitivity grid" in report
    assert "not a validated operational controller" in report
    assert main(["--root", str(paths.root), "report", "--run-id", "fixture-run"]) == 0

    run_manifest_path = run_dir / "run_manifest.json"
    original_run_manifest = run_manifest_path.read_bytes()
    tampered_manifest = json.loads(original_run_manifest)
    tampered_manifest["artifact_schemas"].pop("predictions.csv")
    run_manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(DataError, match="schema declarations"):
        load_finalized_run(run_dir)
    run_manifest_path.write_bytes(original_run_manifest)

    predictions_path = run_dir / "predictions.csv"
    original_predictions = predictions_path.read_bytes()
    prediction_frame = pd.read_csv(predictions_path, parse_dates=["issue_date", "target_date"])
    prediction_frame.loc[0, "issue_date"] -= pd.Timedelta(days=1)
    predictions_path.write_bytes(prediction_frame.to_csv(index=False, lineterminator="\n").encode())
    tampered_manifest = json.loads(original_run_manifest)
    tampered_manifest["artifact_sha256"]["predictions.csv"] = _sha256_file(predictions_path)
    run_manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(DataError, match="differ by one calendar day"):
        load_finalized_run(run_dir)
    predictions_path.write_bytes(original_predictions)
    run_manifest_path.write_bytes(original_run_manifest)

    # Report generation is self-contained: it only needs the finalized run directory.
    paths.raw_path.unlink()
    paths.sidecar_path.unlink()
    paths.daily_path.unlink()
    paths.quality_path.unlink()
    assert render_final_report(run_dir) == report_path

    (second_run.run_dir / "predictions.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing or has changed"):
        load_finalized_run(second_run.run_dir)
    assert _frozen_run_hashes() == frozen_hashes


def test_experiment_finalize_cli_publishes_immutable_report_run(
    tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_hashes = _frozen_run_hashes()
    paths, _ = _build_frozen_fixture(tmp_path)
    config = ExperimentConfig()
    run_experiment_validation(paths, config, "fixture-exp")
    experiment_dir = tmp_path / "artifacts" / "experiments" / "fixture-exp"
    selection_path = experiment_dir / "validation" / "selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bundle_path = experiment_dir / "models" / "ridge_model.json"
    bundle, test_rows, expected = _bundle_test_predictions(paths, bundle_path, selection)
    expected_persistence = np.maximum(
        test_rows["radiation_today_kwh_m2_day"].to_numpy(dtype=float), 0.0
    )
    expected_seasonal = predict_seasonal_mean_bundle(bundle, test_rows).to_numpy()
    digest = _sha256_file(selection_path)

    def forbidden_refit(*_args, **_kwargs):
        pytest.fail("experiment finalization must reuse its local frozen model bundle")

    monkeypatch.setattr(evaluate_module, "fit_ridge", forbidden_refit)
    monkeypatch.setattr(evaluate_module, "baseline_predictions", forbidden_refit)

    from sunreserve.__main__ import main

    args = [
        "--root",
        str(tmp_path),
        "finalize",
        "--manifest-sha256",
        digest,
        "--run-id",
        "experiment-final",
        "--experiment-id",
        "fixture-exp",
    ]
    assert main(args) == 0
    run_dir = paths.runs_dir / "experiment-final"
    run_manifest = load_finalized_run(run_dir)["run_manifest"]
    prediction_frame = pd.read_csv(run_dir / "predictions.csv")
    assert run_manifest["source_experiment_id"] == "fixture-exp"
    assert run_manifest["artifact_type"] == "sunreserve_finalized_experiment"
    assert "test_excluded" not in run_manifest["row_counts"]
    assert run_manifest["excluded_rows_by_split"] == {"train": 6, "validation": 0, "test": 0}
    assert run_manifest["excluded_dates_by_split"]["test"] == []
    assert run_manifest["model_bundle"] == {
        "path": selection["model_bundle"]["path"],
        "model_id": bundle.model_id,
        "sha256": bundle.artifact_sha256,
        "training_split": "train",
        "training_rows": bundle.training_rows,
        "training_target_period": [
            bundle.training_start.isoformat(),
            bundle.training_end.isoformat(),
        ],
    }
    assert prediction_frame["ridge_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected.to_numpy(), abs=1e-10
    )
    assert prediction_frame["persistence_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected_persistence, abs=1e-10
    )
    assert prediction_frame["seasonal_mean_prediction_kwh_m2_day"].to_numpy() == pytest.approx(
        expected_seasonal, abs=1e-10
    )
    assert (run_dir / "selection_manifest.json").read_bytes() == selection_path.read_bytes()
    assert run_manifest["selection_manifest_sha256"] == digest
    assert "Finalized run:" in capsys.readouterr().out
    assert main(["--root", str(tmp_path), "report", "--run-id", "experiment-final"]) == 0
    assert (run_dir / "report.md").is_file()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        finalize_run(
            paths, manifest_sha256=digest, run_id="experiment-final", experiment_id="fixture-exp"
        )
    with pytest.raises(ValueError, match="experiment_id"):
        finalize_run(paths, manifest_sha256=digest, run_id="escape", experiment_id="../outside")
    assert not (tmp_path / "artifacts" / "outside").exists()
    assert _frozen_run_hashes() == frozen_hashes


@pytest.mark.parametrize(
    "failure", ["identity", "configuration", "active_configuration", "digest", "model"]
)
def test_experiment_freeze_rejections_happen_before_test_parse(
    tmp_path, monkeypatch, failure
) -> None:
    paths, _ = _build_frozen_fixture(tmp_path)
    run_experiment_validation(paths, ExperimentConfig(), "checked-exp")
    experiment_dir = tmp_path / "artifacts" / "experiments" / "checked-exp"
    manifest_path = experiment_dir / "experiment_manifest.json"
    experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_path = experiment_dir / "validation" / "selection_manifest.json"
    digest = _sha256_file(selection_path)
    if failure == "identity":
        experiment["run_id"] = "different-exp"
        manifest_path.write_text(json.dumps(experiment), encoding="utf-8")
    elif failure == "configuration":
        experiment["configuration_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(experiment), encoding="utf-8")
    elif failure == "model":
        (experiment_dir / "models" / "ridge_model.json").write_text("tampered", encoding="utf-8")

    def forbidden_test_parse(*_args, **_kwargs):
        pytest.fail("experiment freeze rejection must precede full test CSV parsing")

    monkeypatch.setattr(evaluate_module, "_load_full_daily", forbidden_test_parse)
    supplied_digest = "0" * 64 if failure == "digest" else digest
    expected_config = (
        ExperimentConfig(reserve_buffer_kwh=2.0) if failure == "active_configuration" else None
    )
    error_match = "active configuration" if failure == "active_configuration" else None
    with pytest.raises(DataError, match=error_match):
        finalize_run(
            paths,
            manifest_sha256=supplied_digest,
            run_id=f"reject-{failure}",
            experiment_id="checked-exp",
            expected_config=expected_config,
        )
