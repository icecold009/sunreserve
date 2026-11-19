import hashlib
import json

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig
from sunreserve.data import EXPECTED_UNITS, ProjectPaths, fetch_raw, prepare_daily
from sunreserve.evaluate import load_development_daily, run_validation


def _write_validation_fixture(paths, config, *, sentinel_test_rows: bool = False) -> None:
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
    if sentinel_test_rows:
        prepared = pd.read_csv(paths.daily_path)
        prepared["ALLSKY_SFC_SW_DWN"] = prepared["ALLSKY_SFC_SW_DWN"].astype(object)
        test_rows = pd.to_datetime(prepared["date"]) >= pd.Timestamp(config.test_start)
        prepared.loc[test_rows, "ALLSKY_SFC_SW_DWN"] = "SENTINEL_TEST_ROW"
        prepared.to_csv(paths.daily_path, index=False, lineterminator="\n")
        quality = json.loads(paths.quality_path.read_text(encoding="utf-8"))
        quality["prepared_daily_csv_sha256"] = hashlib.sha256(
            paths.daily_path.read_bytes()
        ).hexdigest()
        paths.quality_path.write_text(json.dumps(quality), encoding="utf-8")


def test_development_loader_reads_only_through_validation_end(tmp_path) -> None:
    config = ExperimentConfig()
    paths = ProjectPaths(tmp_path)
    _write_validation_fixture(paths, config, sentinel_test_rows=True)

    loaded = load_development_daily(paths, config)

    assert len(loaded) == 2557
    assert loaded.index.min() == pd.Timestamp("2016-01-01")
    assert loaded.index.max() == pd.Timestamp("2022-12-31")
    assert loaded["ALLSKY_SFC_SW_DWN"].max() < 5.0


def test_validate_writes_only_training_and_validation_outputs(tmp_path) -> None:
    config = ExperimentConfig(reserve_buffer_kwh=2.5)
    paths = ProjectPaths(tmp_path)
    _write_validation_fixture(paths, config, sentinel_test_rows=True)

    selection = run_validation(paths, config)
    predictions = pd.read_csv(paths.validation_dir / "validation_predictions.csv")
    manifest = json.loads(
        (paths.validation_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (paths.validation_dir / "validation_metrics.json").read_text(encoding="utf-8")
    )
    decision_rows = pd.read_csv(paths.validation_dir / "decision_predictions.csv")

    assert selection.eligible_rows == {"train": 2185, "validation": 365}
    assert set(manifest["eligible_rows"]) == {"train", "validation"}
    assert manifest["status"].startswith("validation_only")
    assert set(manifest["model_selection"]["validation_metrics"]) == {
        "persistence",
        "seasonal_mean",
        "ridge",
    }
    assert manifest["model_selection"]["selected_model"] == selection.selected_model
    assert manifest["model_selection"]["training_only_low_solar_threshold_kwh_m2_day"] >= 0
    assert manifest["model_selection"]["validation_refit"] is False
    assert manifest["model_bundle"]["training_split"] == "train"
    assert manifest["model_bundle"]["training_rows"] == 2185
    assert (
        hashlib.sha256(paths.model_path.read_bytes()).hexdigest()
        == manifest["model_bundle"]["sha256"]
    )
    decision_selection = manifest["decision_selection"]
    assert decision_selection["selection_partition"] == "validation"
    assert decision_selection["selection_target_period"] == [
        "2022-01-01",
        "2022-12-31",
    ]
    assert decision_selection["shortfall_weight"] == 1.0
    assert decision_selection["excess_prepared_reserve_weight"] == 1.0
    assert decision_selection["selected_model"] in {
        "persistence",
        "seasonal_mean",
        "ridge",
    }
    assert set(decision_selection["scores"]) == {
        "persistence",
        "seasonal_mean",
        "ridge",
    }
    assert manifest["decision_simulation"]["primary_scenario"]["buffer_kwh"] == 2.5
    assert metrics["selected_model"] == selection.selected_model
    assert len(predictions) == 365
    assert pd.to_datetime(predictions["target_date"]).min() == pd.Timestamp("2022-01-01")
    assert pd.to_datetime(predictions["target_date"]).max() == pd.Timestamp("2022-12-31")
    assert predictions["target_radiation_kwh_m2_day"].max() < 100.0
    assert manifest["input_hashes"]["prepared_development_rows_sha256"]
    primary_rows = decision_rows.loc[decision_rows["scenario_id"] == "primary"]
    assert set(primary_rows["buffer_kwh"]) == {2.5}
    grid = (
        decision_rows.loc[decision_rows["scenario_id"] != "primary"]
        .groupby("scenario_id", as_index=False)
        .first()
    )
    assert len(grid) == 27
    assert set(grid["demand_kwh"]) == {10.0, 15.0, 20.0}
    assert set(grid["capacity_kwh"]) == {10.0, 15.0, 20.0}
    assert all(
        any(np.isclose(row.buffer_kwh, row.demand_kwh * fraction) for fraction in (0, 0.1, 0.2))
        for row in grid.itertuples()
    )
