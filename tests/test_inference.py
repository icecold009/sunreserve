import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sunreserve.__main__ as cli_module
import sunreserve.inference as inference_module
import sunreserve.prediction_export as prediction_export_module
from sunreserve.__main__ import main as cli_main
from sunreserve.config import ExperimentConfig
from sunreserve.features import (
    FEATURE_COLUMNS,
    RADIATION,
    TARGET_COLUMN,
    build_next_day_features,
    build_supervised_rows,
)
from sunreserve.inference import (
    ModelArtifactError,
    load_model_bundle,
    load_observation_file,
    load_project_model,
    model_bundle_payload,
    observation_frame_from_records,
    predict_next_day,
    predict_ridge_bundle,
    predict_seasonal_mean_bundle,
    write_model_bundle,
)
from sunreserve.observations import OBSERVATION_POLICY_NOTE
from sunreserve.prediction_export import (
    PREDICTION_EXPORT_FIELDS,
    build_prediction_export,
    entrypoint_source_identity,
    generated_at_utc_now,
    package_source_identity,
)
from sunreserve.train import fit_ridge, fit_seasonal_mean, predict_ridge


def train_rows() -> pd.DataFrame:
    count = 80
    rng = np.random.default_rng(42)
    rows = pd.DataFrame(
        {
            "split": "train",
            "issue_date": pd.date_range("2020-01-01", periods=count),
            "target_date": pd.date_range("2020-01-02", periods=count),
            TARGET_COLUMN: np.linspace(1.0, 12.0, count) + rng.normal(0, 0.15, count),
        }
    )
    for position, name in enumerate(FEATURE_COLUMNS):
        rows[name] = 2.0 + np.arange(count) * (0.025 + position * 0.003)
    rows[FEATURE_COLUMNS[0]] = np.linspace(1.0, 10.0, count)
    rows[TARGET_COLUMN] = 0.7 * rows[FEATURE_COLUMNS[0]] + 0.2 * rows[FEATURE_COLUMNS[5]]
    return rows


def config() -> ExperimentConfig:
    return ExperimentConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2021, 12, 31),
        training_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
    )


def observations(start: str = "2024-12-25") -> pd.DataFrame:
    dates = pd.date_range(start, periods=7, freq="D")
    return pd.DataFrame(
        {
            RADIATION: [2.0, 3.0, 4.5, 3.5, 5.0, 6.0, 7.0],
            "T2M": [20.0, 20.5, 21.0, 22.0, 22.5, 23.0, 23.5],
            "RH2M": [55.0, 54.0, 52.0, 50.0, 48.0, 46.0, 45.0],
        },
        index=dates,
    )


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    values = frame.reset_index(names="date")
    values["date"] = values["date"].dt.strftime("%Y-%m-%d")
    return values.to_dict(orient="records")


def test_window_features_match_chronological_builder() -> None:
    frame = observations()
    live = build_next_day_features(frame).iloc[0]
    next_day = frame.index[-1] + pd.Timedelta(days=1)
    next_row = pd.DataFrame({RADIATION: [8.0], "T2M": [24.0], "RH2M": [44.0]}, index=[next_day])
    extended = pd.concat([frame, next_row])
    target_config = ExperimentConfig(
        start_date=date(2024, 12, 25),
        end_date=date(2025, 1, 1),
        training_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 12, 31),
        test_start=date(2023, 1, 1),
    )
    rows, _ = build_supervised_rows(extended, target_config)
    expected = rows.loc[rows["issue_date"] == frame.index[-1]].iloc[0]

    assert live["issue_date"] == frame.index[-1]
    assert live["target_date"] == frame.index[-1] + pd.Timedelta(days=1)
    assert live.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float) == pytest.approx(
        expected.loc[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[:-1], "exactly seven"),
        (lambda frame: frame.drop(frame.index[3]), "exactly seven"),
        (lambda frame: frame.rename(columns={"T2M": "temperature"}), "missing columns"),
        (lambda frame: frame.assign(T2M=np.nan), "finite values"),
        (lambda frame: frame.assign(ALLSKY_SFC_SW_DWN=-1.0), "nonnegative"),
        (lambda frame: frame.assign(RH2M=101.0), "between 0 and 100"),
    ],
)
def test_inference_window_rejects_invalid_fields(mutate, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_next_day_features(mutate(observations()))


def test_inference_window_rejects_gaps_duplicates_and_reversed_dates() -> None:
    frame = observations()
    with pytest.raises(ValueError, match="consecutive"):
        offsets = pd.to_timedelta([0, 1, 2, 3, 4, 6, 7], unit="D")
        build_next_day_features(frame.set_axis(frame.index + offsets))
    with pytest.raises(ValueError, match="duplicate"):
        build_next_day_features(frame.set_axis([*frame.index[:6], frame.index[5]]))
    with pytest.raises(ValueError, match="chronological"):
        build_next_day_features(frame.iloc[::-1])


def test_inference_window_rejects_future_observations() -> None:
    tomorrow = pd.Timestamp(date.today() + pd.Timedelta(days=1))
    future = observations(start=(tomorrow - pd.Timedelta(days=6)).date().isoformat())

    with pytest.raises(ValueError, match="future-dated"):
        build_next_day_features(future)


def test_json_observation_contract_rejects_missing_and_extra_fields() -> None:
    source = records(observations())
    missing = [dict(row) for row in source]
    missing[0].pop("RH2M")
    with pytest.raises(ValueError, match="missing fields"):
        observation_frame_from_records(missing)
    extra = [dict(row) for row in source]
    extra[0]["target"] = 100.0
    with pytest.raises(ValueError, match="unsupported fields"):
        observation_frame_from_records(extra)
    boolean = [dict(row) for row in source]
    boolean[0][RADIATION] = True
    with pytest.raises(ValueError, match="must be numeric"):
        observation_frame_from_records(boolean)
    invalid_date = [dict(row) for row in source]
    invalid_date[0]["date"] = "20241225"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        observation_frame_from_records(invalid_date)


def test_model_bundle_is_train_only_and_matches_sklearn_pipeline(tmp_path) -> None:
    train = train_rows()
    model, digest = write_model_bundle(train, config(), tmp_path / "ridge.json")
    loaded = load_model_bundle(tmp_path / "ridge.json")
    live_features = build_next_day_features(observations())
    direct = predict_ridge(fit_ridge(train, config()), live_features).iloc[0]
    actual = predict_next_day(loaded, observations())

    assert model.artifact_sha256 == digest
    assert loaded.artifact_sha256 == digest
    assert loaded.training_rows == len(train)
    assert loaded.training_start == pd.Timestamp(train["target_date"].min()).date()
    assert loaded.training_end == pd.Timestamp(train["target_date"].max()).date()
    assert actual.radiation_kwh_m2_day == pytest.approx(direct, abs=1e-10)
    assert actual.target_date == date(2025, 1, 1)


def test_vectorized_bundle_predictions_match_fitted_sklearn_pipeline(tmp_path) -> None:
    training = train_rows()
    evaluation = training.copy()
    evaluation.loc[evaluation.index[::11], FEATURE_COLUMNS[2]] = np.nan
    fitted_pipeline = fit_ridge(training, config())
    bundle, _ = write_model_bundle(training, config(), tmp_path / "ridge.json")

    expected = predict_ridge(fitted_pipeline, evaluation)
    actual = predict_ridge_bundle(bundle, evaluation)

    assert actual.index.equals(evaluation.index)
    assert actual.name == "ridge_prediction_kwh_m2_day"
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-10)


def test_vectorized_bundle_predictions_reject_nonfinite_and_overflow_results(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    evaluation = train_rows().iloc[:4].copy()

    nonfinite_input = evaluation.copy()
    nonfinite_input.iloc[0, nonfinite_input.columns.get_loc(FEATURE_COLUMNS[0])] = np.inf
    with pytest.raises(ValueError, match="finite or missing for imputation"):
        predict_ridge_bundle(bundle, nonfinite_input)

    tiny_scales = bundle.scaler_scale.copy()
    tiny_scales[0] = 1e-320
    overflow_scaled = replace(bundle, scaler_scale=tiny_scales)
    with pytest.raises(ValueError, match="scaled inference features became non-finite"):
        predict_ridge_bundle(overflow_scaled, evaluation)

    huge_coefficients = np.zeros_like(bundle.coefficients)
    huge_coefficients[0] = 1e308
    overflow_prediction = replace(bundle, coefficients=huge_coefficients)
    extreme_rows = evaluation.copy()
    extreme_rows.iloc[0, extreme_rows.columns.get_loc(FEATURE_COLUMNS[0])] = 1e308
    with pytest.raises(ValueError, match="Ridge inference produced a non-finite forecast"):
        predict_ridge_bundle(overflow_prediction, extreme_rows)

    overflow_intercept = replace(bundle, intercept=10**400)
    with pytest.raises(ValueError, match="intercept must be finite"):
        predict_ridge_bundle(overflow_intercept, evaluation)


def test_seasonal_bundle_predictions_match_training_baseline_with_global_fallback(tmp_path) -> None:
    training = train_rows()
    bundle, _ = write_model_bundle(training, config(), tmp_path / "ridge.json")
    evaluation = training.iloc[:3].copy()
    evaluation["target_date"] = pd.to_datetime(["2020-01-10", "2020-05-10", "2020-12-10"])

    expected = fit_seasonal_mean(training).predict(evaluation)
    actual = predict_seasonal_mean_bundle(bundle, evaluation)

    assert actual.index.equals(evaluation.index)
    assert actual.name == "seasonal_mean_prediction_kwh_m2_day"
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-10)
    assert actual.iloc[1] == pytest.approx(bundle.training_global_mean)
    assert actual.iloc[2] == pytest.approx(bundle.training_global_mean)


@pytest.mark.parametrize(
    ("dates", "message"),
    [
        (["not-a-date"], "valid calendar dates"),
        (["2020-01-01 12:00:00"], "timezone-naive daily dates"),
        (["2020-01-01T00:00:00+00:00"], "timezone-naive daily dates"),
        (["2020-01-01", "2020-01-01"], "valid and unique"),
    ],
)
def test_seasonal_bundle_predictions_reject_invalid_dates(tmp_path, dates, message) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    rows = pd.DataFrame({"target_date": dates})

    with pytest.raises(ValueError, match=message):
        predict_seasonal_mean_bundle(bundle, rows)


def test_seasonal_bundle_predictions_reject_nonfinite_means(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    rows = pd.DataFrame({"target_date": pd.to_datetime(["2020-01-01"])})
    nonfinite_mean = replace(bundle, monthly_means={1: np.inf})

    with pytest.raises(ValueError, match="means must be finite"):
        predict_seasonal_mean_bundle(nonfinite_mean, rows)


def test_model_export_rejects_mixed_or_post_training_rows() -> None:
    mixed = train_rows()
    mixed.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="training partition"):
        model_bundle_payload(mixed, config())
    future = train_rows()
    future["target_date"] = pd.date_range("2022-01-01", periods=len(future))
    with pytest.raises(ValueError, match="extend beyond"):
        model_bundle_payload(future, config())


def test_predictions_change_when_valid_latest_input_changes(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    first = observations()
    changed = first.copy()
    changed.iloc[-1, changed.columns.get_loc(RADIATION)] += 2.0

    first_result = predict_next_day(bundle, first)
    changed_result = predict_next_day(bundle, changed)

    assert first_result.radiation_kwh_m2_day != pytest.approx(
        changed_result.radiation_kwh_m2_day, abs=1e-6
    )
    assert first_result.persistence_kwh_m2_day != changed_result.persistence_kwh_m2_day


def test_plausible_unusual_values_remain_unclipped_and_report_policy(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    unusual = observations()
    unusual["T2M"] = 60.0

    result = predict_next_day(bundle, unusual)

    assert result.features["temperature_today_c"] == 60.0
    assert result.warnings == (OBSERVATION_POLICY_NOTE,)


def test_inference_rejects_nonfinite_scaled_features(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    scales = bundle.scaler_scale.copy()
    scales[0] = 1e-320
    extreme = replace(bundle, scaler_scale=scales)

    with pytest.raises(ValueError, match="scaled inference features became non-finite"):
        predict_next_day(extreme, observations())


def test_inference_rejects_nonfinite_ridge_and_baseline_outputs(tmp_path) -> None:
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    huge_observations = observations()
    huge_observations.iloc[-1, huge_observations.columns.get_loc(RADIATION)] = 1e308
    coefficients = np.zeros_like(bundle.coefficients)
    coefficients[0] = 1e308
    extreme_ridge = replace(bundle, coefficients=coefficients)

    with pytest.raises(ValueError, match="Ridge inference produced a non-finite forecast"):
        predict_next_day(extreme_ridge, huge_observations)

    nonfinite_baseline = replace(bundle, monthly_means={1: float("inf")})
    with pytest.raises(ValueError, match="seasonal baseline produced a non-finite forecast"):
        predict_next_day(nonfinite_baseline, observations())


def test_missing_or_corrupt_model_fails_clearly(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="missing or unreadable"):
        load_model_bundle(tmp_path / "missing.json")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="corrupt"):
        load_model_bundle(corrupt)

    malformed = model_bundle_payload(train_rows(), config())
    malformed["feature_columns"] = None
    corrupt.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="list of strings"):
        load_model_bundle(corrupt)
    malformed["feature_columns"] = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    corrupt.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="list of strings"):
        load_model_bundle(corrupt)
    malformed["feature_columns"] = list(FEATURE_COLUMNS)
    malformed["preprocessing"]["scaler_scale"][0] = 0.0
    corrupt.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="scales must be positive"):
        load_model_bundle(corrupt)


def test_project_model_must_match_validation_manifest_hash(tmp_path) -> None:
    model_path = tmp_path / "models" / "ridge_model.json"
    bundle, digest = write_model_bundle(train_rows(), config(), model_path)
    manifest_path = tmp_path / "artifacts" / "validation" / "selection_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "status": "validation_only; final test not loaded",
                "model_bundle": {
                    "path": "models/ridge_model.json",
                    "sha256": digest,
                    "model_id": bundle.model_id,
                    "training_split": "train",
                    "training_rows": bundle.training_rows,
                    "training_target_period": [
                        bundle.training_start.isoformat(),
                        bundle.training_end.isoformat(),
                    ],
                },
                "decision_selection": {"selected_model": "ridge"},
            }
        ),
        encoding="utf-8",
    )

    loaded, decision_selection = load_project_model(tmp_path)

    assert loaded.artifact_sha256 == digest
    assert decision_selection["selected_model"] == "ridge"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(digest, "0" * 64),
        encoding="utf-8",
    )
    with pytest.raises(ModelArtifactError, match="SHA-256"):
        load_project_model(tmp_path)


def test_project_model_can_load_selected_isolated_experiment(tmp_path) -> None:
    experiment_id = "trial-1"
    experiment_dir = tmp_path / "artifacts" / "experiments" / experiment_id
    validation_dir = experiment_dir / "validation"
    model_path = experiment_dir / "models" / "ridge_model.json"
    validation_dir.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    bundle, model_digest = write_model_bundle(train_rows(), config(), model_path)
    selection = {
        "status": "validation_only; final test not loaded",
        "model_bundle": {
            "path": f"artifacts/experiments/{experiment_id}/models/ridge_model.json",
            "sha256": model_digest,
            "model_id": bundle.model_id,
            "training_split": "train",
            "training_rows": bundle.training_rows,
            "training_target_period": [
                bundle.training_start.isoformat(),
                bundle.training_end.isoformat(),
            ],
        },
        "decision_selection": {"selected_model": "ridge"},
    }
    selection_path = validation_dir / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "sunreserve_validation_experiment",
                "run_id": experiment_id,
                "model_path": f"artifacts/experiments/{experiment_id}/models/ridge_model.json",
                "model_sha256": model_digest,
                "selection_manifest_sha256": hashlib.sha256(
                    selection_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    loaded, decision_selection = load_project_model(tmp_path, experiment_id=experiment_id)

    assert loaded.artifact_sha256 == model_digest
    assert decision_selection["selected_model"] == "ridge"
    assert not (tmp_path / "models" / "ridge_model.json").exists()

    selection_path.write_text(json.dumps({**selection, "changed": True}), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="do not match their manifest"):
        load_project_model(tmp_path, experiment_id=experiment_id)


def test_prediction_export_uses_identity_captured_at_inference(monkeypatch, tmp_path) -> None:
    old_identity = {
        "identity_type": "installed_package_python_sources",
        "algorithm": "sha256",
        "digest": "a" * 64,
        "file_count": 7,
    }
    reloaded_identity = {**old_identity, "digest": "b" * 64}
    bundle, _ = write_model_bundle(train_rows(), config(), tmp_path / "ridge.json")
    monkeypatch.setattr(inference_module, "package_source_identity", lambda: old_identity)
    monkeypatch.setattr(inference_module, "version", lambda _name: "0.1.0-producer")

    prediction = predict_next_day(bundle, observations())

    monkeypatch.setattr(
        prediction_export_module,
        "package_source_identity",
        lambda: reloaded_identity,
    )
    reserve_outputs = [
        {
            "approach": approach,
            "demand_kwh": 15.0,
            "predicted_generation_proxy_kwh_day": 2.0,
            "prepared_reserve_kwh": 1.0,
            "buffer_kwh": 2.0,
        }
        for approach in ("ridge", "persistence", "seasonal_mean")
    ]
    exported = build_prediction_export(
        generated_at_utc=generated_at_utc_now(),
        issue_date=prediction.issue_date,
        target_date=prediction.target_date,
        source_kind="test_observations",
        observations=observations(),
        assumptions={
            "daily_demand_kwh": 15.0,
            "reserve_capacity_kwh": 10.0,
            "reserve_buffer_kwh": 2.0,
            "panel_area_m2": 20.0,
            "conversion_efficiency": 0.2,
            "derating_factor": 0.8,
        },
        model_identity={
            "model_id": bundle.model_id,
            "artifact_sha256": bundle.artifact_sha256,
            "training_rows": bundle.training_rows,
            "training_start": bundle.training_start,
            "training_end": bundle.training_end,
        },
        package_version=prediction.package_version,
        package_source=prediction.package_source_identity,
        entrypoint_identity=entrypoint_source_identity("cli", b"entrypoint"),
        forecasts={
            "ridge_radiation_kwh_m2_day": prediction.radiation_kwh_m2_day,
            "persistence_radiation_kwh_m2_day": prediction.persistence_kwh_m2_day,
            "seasonal_mean_radiation_kwh_m2_day": prediction.seasonal_mean_kwh_m2_day,
        },
        reserve_outputs=reserve_outputs,
        features=prediction.features,
        warnings=prediction.warnings,
        input_validation_policy=OBSERVATION_POLICY_NOTE,
        result_status="fresh",
    )

    assert prediction.package_version == "0.1.0-producer"
    assert exported["package_source_identity"] == old_identity
    assert exported["package_source_identity"] != reloaded_identity


def test_observation_json_file_is_validated(tmp_path) -> None:
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"observations": records(observations())}), encoding="utf-8")
    loaded = load_observation_file(path)
    assert len(loaded) == 7
    assert loaded.index.is_monotonic_increasing
    path.write_text(json.dumps({"observations": records(observations())[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seven"):
        load_observation_file(path)
    invalid = records(observations())
    invalid[2]["T2M"] = -999
    path.write_text(json.dumps({"observations": invalid}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"observation 3 field T2M.*NASA POWER -999"):
        load_observation_file(path)


def test_predict_cli_uses_real_model_and_emits_reserve_comparison(tmp_path, capsys) -> None:
    model_path = tmp_path / "ridge.json"
    model, _ = write_model_bundle(train_rows(), config(), model_path)
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(
        json.dumps({"observations": records(observations())}), encoding="utf-8"
    )

    result = cli_main(
        [
            "--root",
            str(tmp_path),
            "predict",
            "--observations",
            str(observations_path),
            "--model",
            str(model_path),
        ]
    )
    stdout = capsys.readouterr().out
    output = json.loads(stdout)

    assert result == 0
    assert set(output) == PREDICTION_EXPORT_FIELDS
    assert output["format"] == "sunreserve-prediction-export"
    assert output["schema_version"] == 1
    assert output["result_status"] == "not_applicable"
    assert output["issue_date"] == "2024-12-31"
    assert output["target_date"] == "2025-01-01"
    assert output["source"]["kind"] == "cli_observation_file"
    assert output["source"]["availability_status"] == "unverified"
    assert output["model_identity"]["model_id"] == "sunreserve.ridge.train-only.v1"
    assert output["model_identity"]["artifact_sha256"]
    assert output["model_identity"]["training_period"] == {
        "start_date": model.training_start.isoformat(),
        "end_date": model.training_end.isoformat(),
    }
    assert output["package_version"] == "0.1.0"
    assert output["package_source_identity"] == package_source_identity()
    assert output["entrypoint_identity"] == {
        "application_kind": "cli",
        "algorithm": "sha256",
        "digest": hashlib.sha256(Path(cli_module.__file__).read_bytes()).hexdigest(),
    }
    assert output["input_validation_policy"] == OBSERVATION_POLICY_NOTE
    assert output["prediction"]["warnings"] == [OBSERVATION_POLICY_NOTE]
    assert len(output["observations"]) == 7
    assert output["assumptions"]["daily_demand_kwh"] == {
        "value": 15.0,
        "unit": "kWh/day",
    }
    assert len(output["reserve_outputs"]) == 3
    assert all("prepared_reserve_kwh" in row for row in output["reserve_outputs"])
    assert all("backup_need_before_capacity_cap_kwh" in row for row in output["reserve_outputs"])
    assert str(tmp_path) not in stdout
    json.dumps(output, allow_nan=False)
