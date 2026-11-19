from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sunreserve.evaluate as evaluate
from sunreserve.config import ExperimentConfig
from sunreserve.data import EXPECTED_UNITS, DataError, ProjectPaths, fetch_raw, prepare_daily
from sunreserve.inference import load_model_bundle


def _write_validation_inputs(paths, config):
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    position = np.arange(len(dates), dtype=float)
    date_keys = dates.strftime("%Y%m%d")
    values = {
        "ALLSKY_SFC_SW_DWN": 3.5 + np.sin(position / 16) + 0.2 * np.cos(position / 3),
        "T2M": 23.0 + np.sin(position / 27),
        "RH2M": 60.0 + 10 * np.cos(position / 19),
    }
    payload = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(config.longitude, 3), round(config.latitude, 3), 840.0],
        },
        "header": {
            "start": config.start_date.strftime("%Y%m%d"),
            "end": config.end_date.strftime("%Y%m%d"),
            "time_standard": config.time_standard,
        },
        "parameters": {name: {"units": EXPECTED_UNITS[name][0]} for name in config.parameters},
        "properties": {
            "parameter": {
                name: dict(zip(date_keys, values[name].tolist(), strict=True))
                for name in config.parameters
            }
        },
    }
    fetch_raw(
        config,
        paths,
        downloader=lambda _url: json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )
    prepare_daily(config, paths)


def fake_validation(
    paths,
    _config,
    *,
    output_dir=None,
    model_path=None,
    recorded_model_path=None,
    _verified_daily=None,
):
    assert output_dir is not None and model_path is not None and recorded_model_path is not None
    assert _verified_daily is not None
    assert _verified_daily.index.max() <= pd.Timestamp(_config.validation_end)
    (output_dir / "selection_manifest.json").write_text(
        json.dumps({"decision_selection": {"selected_model": "ridge"}}), encoding="utf-8"
    )
    model_path.write_text("temporary model", encoding="utf-8")
    return SimpleNamespace(selected_model="ridge")


def test_experiment_validation_publishes_only_scoped_outputs(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, ExperimentConfig())
    monkeypatch.setattr(evaluate, "run_validation", fake_validation)
    selection, model_path = evaluate.run_experiment_validation(paths, ExperimentConfig(), "trial-1")
    run_dir = tmp_path / "artifacts" / "experiments" / "trial-1"
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert selection.selected_model == "ridge"
    assert model_path == run_dir / "models" / "ridge_model.json"
    assert manifest["model_path"] == "artifacts/experiments/trial-1/models/ridge_model.json"
    assert manifest["model_sha256"] == evaluate._sha256_file(model_path)
    assert manifest["selection_manifest_sha256"] == evaluate._sha256_file(
        run_dir / "validation" / "selection_manifest.json"
    )
    assert not paths.validation_dir.exists()
    assert not paths.model_path.exists()


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", ".hidden", "x" * 65])
def test_experiment_id_must_be_safe(tmp_path, run_id):
    with pytest.raises(ValueError, match="--run-id"):
        evaluate.run_experiment_validation(ProjectPaths(tmp_path), ExperimentConfig(), run_id)


def test_duplicate_and_reserved_ids_fail_without_overwrite(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, ExperimentConfig())
    monkeypatch.setattr(evaluate, "run_validation", fake_validation)
    _, model = evaluate.run_experiment_validation(paths, ExperimentConfig(), "trial")
    before = model.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        evaluate.run_experiment_validation(paths, ExperimentConfig(), "trial")
    assert model.read_bytes() == before
    lock = tmp_path / "artifacts" / "experiments" / ".locks" / "busy.lock"
    lock.mkdir()
    with pytest.raises(FileExistsError, match="active or stale"):
        evaluate.run_experiment_validation(paths, ExperimentConfig(), "busy")
    assert lock.is_dir()


def test_concurrent_same_id_is_reserved_exclusively(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, ExperimentConfig())
    started = threading.Event()
    release = threading.Event()

    def slow_validation(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return fake_validation(*args, **kwargs)

    monkeypatch.setattr(evaluate, "run_validation", slow_validation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(evaluate.run_experiment_validation, paths, ExperimentConfig(), "same")
        assert started.wait(5)
        with pytest.raises(FileExistsError, match="active or stale"):
            evaluate.run_experiment_validation(paths, ExperimentConfig(), "same")
        release.set()
        pending.result(timeout=5)


def test_failure_cleans_staging_and_owned_reservation(tmp_path, monkeypatch):
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, ExperimentConfig())

    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(evaluate, "run_validation", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        evaluate.run_experiment_validation(paths, ExperimentConfig(), "failed")
    experiments = tmp_path / "artifacts" / "experiments"
    assert not list(experiments.glob(".failed-*"))
    assert not (experiments / ".locks" / "failed.lock").exists()
    assert not paths.validation_dir.exists()
    assert not paths.model_path.exists()


@pytest.mark.parametrize(
    "failure",
    [
        "prepared_bytes",
        "raw_bytes",
        "raw_hash",
        "source",
        "units",
        "location",
        "schema",
        "configuration",
        "missing_field",
    ],
)
def test_experiment_provenance_preflight_fails_before_creating_outputs(tmp_path, failure):
    config = ExperimentConfig()
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, config)
    if failure == "prepared_bytes":
        paths.daily_path.write_bytes(paths.daily_path.read_bytes() + b"\nchanged\n")
    elif failure == "raw_bytes":
        paths.raw_path.write_bytes(paths.raw_path.read_bytes() + b" ")
    else:
        quality = json.loads(paths.quality_path.read_text(encoding="utf-8"))
        sidecar = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
        if failure == "raw_hash":
            sidecar["sha256"] = "0" * 64
        elif failure == "source":
            quality["source"] = "different-source"
        elif failure == "units":
            sidecar["units"]["T2M"] = "F"
        elif failure == "location":
            sidecar["reported_location"]["latitude"] += 0.01
        elif failure == "schema":
            quality["schema_version"] = 999
        elif failure == "configuration":
            sidecar["request_parameters"]["parameters"] = ["T2M"]
        elif failure == "missing_field":
            del quality["calendar_days"]
        paths.quality_path.write_text(json.dumps(quality), encoding="utf-8")
        paths.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(DataError):
        evaluate.run_experiment_validation(paths, config, "bad-provenance")

    assert not (tmp_path / "artifacts").exists()
    assert not paths.validation_dir.exists()
    assert not paths.model_path.exists()


def test_cli_validate_publishes_complete_temporary_experiment(tmp_path, capsys, monkeypatch):
    from sunreserve.__main__ import main

    config = ExperimentConfig()
    paths = ProjectPaths(tmp_path)
    _write_validation_inputs(paths, config)
    prepared = pd.read_csv(paths.daily_path)
    prepared["ALLSKY_SFC_SW_DWN"] = prepared["ALLSKY_SFC_SW_DWN"].astype(object)
    prepared.loc[len(prepared) - 1, "ALLSKY_SFC_SW_DWN"] = "SENTINEL_TEST_ROW"
    prepared.to_csv(paths.daily_path, index=False, lineterminator="\n")
    quality = json.loads(paths.quality_path.read_text(encoding="utf-8"))
    quality["prepared_daily_csv_sha256"] = hashlib.sha256(paths.daily_path.read_bytes()).hexdigest()
    paths.quality_path.write_text(json.dumps(quality), encoding="utf-8")

    assert main(["--root", str(tmp_path), "validate", "--run-id", "cli-fixture"]) == 0
    output = capsys.readouterr().out
    run_dir = tmp_path / "artifacts" / "experiments" / "cli-fixture"
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (run_dir / "validation" / "selection_manifest.json").read_text(encoding="utf-8")
    )
    model_path = tmp_path / manifest["model_path"]
    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == "sunreserve_validation_experiment"
    assert manifest["code_revision"] == "unavailable" or len(manifest["code_revision"]) in (40, 64)
    assert manifest["working_tree_dirty"] in (True, False, None)
    assert selection["code_revision"] == "unavailable" or len(selection["code_revision"]) in (
        40,
        64,
    )
    assert selection["working_tree_dirty"] in (True, False, None)
    assert "source_provenance" in selection["code_hashes"]
    assert selection["eligible_rows"] == {"train": 2185, "validation": 365}
    assert "Training-only inference model:" in output and str(model_path) in output
    assert load_model_bundle(model_path).model_id
    assert {"config.json", "experiment_manifest.json", "validation", "models"} <= {
        path.name for path in run_dir.iterdir()
    }
    assert not paths.validation_dir.exists()
    assert not paths.model_path.exists()

    class LaterDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 19, tzinfo=tz or UTC)

    monkeypatch.setattr(evaluate, "datetime", LaterDateTime)
    monkeypatch.setattr(
        evaluate,
        "source_provenance",
        lambda: {"code_revision": "b2" * 20, "working_tree_dirty": False},
    )
    _, second_model_path = evaluate.run_experiment_validation(paths, config, "cli-fixture-later")
    second_dir = second_model_path.parents[1]
    second_manifest = json.loads(
        (second_dir / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    second_selection = json.loads(
        (second_dir / "validation" / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert second_manifest["created_utc"] != manifest["created_utc"]
    assert second_manifest["code_revision"] != manifest["code_revision"]
    assert second_manifest["working_tree_dirty"] is False
    assert second_manifest["configuration_sha256"] == manifest["configuration_sha256"]
    assert second_manifest["model_sha256"] == manifest["model_sha256"]
    assert second_selection["configuration_sha256"] == selection["configuration_sha256"]
    assert second_selection["input_hashes"] == selection["input_hashes"]
    assert second_selection["model_bundle"]["sha256"] == selection["model_bundle"]["sha256"]
    assert second_selection["created_utc"] != selection["created_utc"]
