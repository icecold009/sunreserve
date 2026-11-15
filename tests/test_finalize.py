from __future__ import annotations

import json
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
    run_validation,
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


def test_offline_prepare_validate_finalize_report_contract(tmp_path, monkeypatch) -> None:
    paths, frozen_digest = _build_frozen_fixture(tmp_path)
    manifest_path = paths.validation_dir / "selection_manifest.json"
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert selection["eligible_rows"] == {"train": 2185, "validation": 365}
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
        "test_excluded": 6,
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
    assert run_manifest["code_revision"] == "uncommitted"
    assert run_manifest["public_demo_bundle"]["status"] == "not_created"
    assert run_manifest["row_counts"]["test"] == 731

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
