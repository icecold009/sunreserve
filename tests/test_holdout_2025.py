from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sunreserve import holdout_2025 as holdout_module
from sunreserve.config import ExperimentConfig
from sunreserve.data import (
    DataError,
    ProjectPaths,
    fetch_raw,
    parse_power_payload,
    prepare_daily,
    request_url,
)
from sunreserve.evaluate import _sha256_file, run_validation
from sunreserve.holdout_2025 import evaluate_2025_holdout
from sunreserve.holdout_2025_cli import main as holdout_cli_main


def _power_payload(config: ExperimentConfig, *, holdout: bool = False) -> bytes:
    dates = pd.date_range(config.start_date, config.end_date, freq="D")
    position = np.arange(len(dates), dtype=float)
    annual = np.sin(position / 365 * 2 * np.pi)
    daily = np.cos(position / 13)
    radiation = 4.1 + 0.65 * annual + 0.22 * daily
    temperature = 22.0 + 3.0 * annual
    humidity = 58.0 - 11.0 * annual
    values: dict[str, dict[str, float | None]] = {
        "ALLSKY_SFC_SW_DWN": dict(zip(dates.strftime("%Y%m%d"), radiation, strict=True)),
        "T2M": dict(zip(dates.strftime("%Y%m%d"), temperature, strict=True)),
        "RH2M": dict(zip(dates.strftime("%Y%m%d"), humidity, strict=True)),
    }
    if holdout:
        # Synthetic missingness tests common-date filtering without touching real outcomes.
        values["ALLSKY_SFC_SW_DWN"]["20250110"] = None
        values["RH2M"]["20250203"] = None
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
        "properties": {"parameter": values},
    }
    return json.dumps(document, separators=(",", ":")).encode()


@pytest.fixture(scope="module")
def frozen_synthetic_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    root = tmp_path_factory.mktemp("sunreserve-holdout")
    paths = ProjectPaths(root)
    config = ExperimentConfig()
    fetch_raw(config, paths, downloader=lambda _url: _power_payload(config))
    prepare_daily(config, paths)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "assumptions.md").write_text(
        "Synthetic fixture only; assumptions are hypothetical.\n", encoding="utf-8"
    )
    run_validation(paths, config)

    holdout_config = ExperimentConfig(
        start_date=pd.Timestamp("2025-01-01").date(),
        end_date=pd.Timestamp("2025-12-31").date(),
    )
    raw = _power_payload(holdout_config, holdout=True)
    _, parsed_metadata = parse_power_payload(raw, holdout_config)
    holdout_path = root / "data" / "raw" / "power_daily_sunreserve_2025_holdout.json"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_bytes(raw)
    parameters = {name: item["units"] for name, item in parsed_metadata["parameters"].items()}
    sidecar = {
        "schema_version": 2,
        "source": config.source,
        "request_url": request_url(holdout_config),
        "request_parameters": {
            "latitude": config.latitude,
            "longitude": config.longitude,
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "parameters": list(config.parameters),
            "community": config.community,
            "time_standard": "UTC",
            "format": "JSON",
        },
        "retrieved_utc": datetime(2026, 1, 2, tzinfo=UTC).isoformat(),
        "reported_location": parsed_metadata["reported_location"],
        "units": parameters,
        "missing_value_marker": -999.0,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    holdout_path.with_suffix(".meta.json").write_text(
        json.dumps(sidecar, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = paths.validation_dir / "selection_manifest.json"
    return (
        root,
        _sha256_file(manifest_path),
        _sha256_file(paths.model_path),
    )


@pytest.fixture(autouse=True)
def use_synthetic_frozen_digests(
    monkeypatch: pytest.MonkeyPatch,
    frozen_synthetic_project: tuple[Path, str, str],
) -> None:
    _, manifest_hash, model_hash = frozen_synthetic_project
    monkeypatch.setattr(holdout_module, "FROZEN_SELECTION_SHA256", manifest_hash)
    monkeypatch.setattr(holdout_module, "FROZEN_MODEL_SHA256", model_hash)


def test_synthetic_holdout_cli_contract_and_artifact_integrity(
    frozen_synthetic_project: tuple[Path, str, str],
) -> None:
    root, manifest_hash, model_hash = frozen_synthetic_project
    result = evaluate_2025_holdout(
        root,
        manifest_sha256=manifest_hash,
        model_sha256=model_hash,
        run_id="fixture-holdout-2025",
    )
    run_dir = result.run_dir
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_revision"] == "unavailable" or len(manifest["code_revision"]) in (40, 64)
    assert manifest["working_tree_dirty"] in (True, False, None)
    assert "source_provenance" in manifest["holdout_evaluation_code_hashes"]
    predictions = pd.read_csv(
        run_dir / "predictions.csv", parse_dates=["issue_date", "target_date"]
    )
    assert manifest["partition"] == "untouched_2025_holdout"
    assert manifest["row_counts"]["2023_2024_targets_scored"] == 0
    assert manifest["no_2025_tuning_or_selection"] is True
    assert set(predictions["target_date"].dt.year) == {2025}
    assert (predictions["issue_date"] == predictions["target_date"] - pd.Timedelta(days=1)).all()
    assert predictions["target_date"].is_unique
    assert set(predictions["split"]) == {"untouched_2025_holdout"}
    quality = json.loads((run_dir / "data_quality.json").read_text(encoding="utf-8"))
    assert quality["missing_counts_by_variable"]["ALLSKY_SFC_SW_DWN"] == 1
    assert quality["missing_counts_by_variable"]["RH2M"] == 1
    excluded_target_dates = {row["target_date"] for row in quality["excluded_targets"]}
    assert {"2025-01-10", "2025-01-11", "2025-01-17"} <= excluded_target_dates
    assert {"ridge", "persistence", "seasonal_mean"} == set(
        json.loads((run_dir / "forecast_metrics.json").read_text())["metrics"]
    )

    decision_predictions = pd.read_csv(
        run_dir / "decision_predictions.csv", parse_dates=["target_date"]
    )
    decision_metrics = pd.read_csv(run_dir / "decision_metrics.csv")
    assert set(decision_metrics["approach"]) == {"ridge", "persistence", "seasonal_mean"}
    assert set(decision_metrics["sample_count"]) == {len(predictions)}
    assert (
        decision_predictions.groupby("approach")["target_date"]
        .apply(lambda dates: dates.reset_index(drop=True).equals(predictions["target_date"]))
        .all()
    )
    for _, row in decision_metrics.iterrows():
        subset = decision_predictions[decision_predictions["approach"] == row["approach"]]
        expected_score = (
            subset["simulated_shortfall_kwh"].mean() + subset["excess_prepared_reserve_kwh"].mean()
        )
        assert row["mean_shortfall_plus_mean_excess_kwh_day"] == pytest.approx(expected_score)
        assert row["shortfall_weight"] == row["excess_prepared_reserve_weight"] == 1.0
    for name, expected_hash in manifest["artifact_sha256"].items():
        assert _sha256_file(run_dir / name) == expected_hash
    frozen_selection = json.loads((run_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["artifact_schemas"]["selection_manifest.json"]
        == frozen_selection["schema_version"]
    )
    assert manifest["artifact_sha256"]["selection_manifest.json"] == _sha256_file(
        run_dir / "selection_manifest.json"
    )
    source_meta_path = root / manifest["source_snapshot"]["provenance_path"]
    assert manifest["source_snapshot"]["provenance_sha256"] == _sha256_file(source_meta_path)

    with pytest.raises(DataError, match="already exists"):
        evaluate_2025_holdout(
            root,
            manifest_sha256=manifest_hash,
            model_sha256=model_hash,
            run_id="fixture-holdout-2025",
        )


def test_wrong_freeze_and_model_digest_fail_before_holdout_read(
    frozen_synthetic_project: tuple[Path, str, str],
) -> None:
    root, manifest_hash, model_hash = frozen_synthetic_project
    holdout_path = root / "data" / "raw" / "power_daily_sunreserve_2025_holdout.json"
    original = holdout_path.read_bytes()
    holdout_path.unlink()
    try:
        with pytest.raises(DataError, match="selection manifest digest"):
            evaluate_2025_holdout(
                root,
                manifest_sha256="0" * 64,
                model_sha256=model_hash,
                run_id="wrong-manifest",
            )
        with pytest.raises(DataError, match="model digest"):
            evaluate_2025_holdout(
                root,
                manifest_sha256=manifest_hash,
                model_sha256="0" * 64,
                run_id="wrong-model",
            )
        assert (
            holdout_cli_main(
                [
                    "--root",
                    str(root),
                    "evaluate-2025",
                    "--manifest-sha256",
                    "0" * 64,
                    "--model-sha256",
                    model_hash,
                    "--run-id",
                    "cli-wrong-manifest",
                ]
            )
            == 2
        )
    finally:
        holdout_path.write_bytes(original)
