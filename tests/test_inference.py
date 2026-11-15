import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

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
    write_model_bundle,
)
from sunreserve.train import fit_ridge, predict_ridge


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


def test_observation_json_file_is_validated(tmp_path) -> None:
    path = tmp_path / "observations.json"
    path.write_text(json.dumps({"observations": records(observations())}), encoding="utf-8")
    loaded = load_observation_file(path)
    assert len(loaded) == 7
    assert loaded.index.is_monotonic_increasing
    path.write_text(json.dumps({"observations": records(observations())[:-1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seven"):
        load_observation_file(path)


def test_predict_cli_uses_real_model_and_emits_reserve_comparison(tmp_path, capsys) -> None:
    model_path = tmp_path / "ridge.json"
    write_model_bundle(train_rows(), config(), model_path)
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
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert output["target_date"] == "2025-01-01"
    assert output["model_id"] == "sunreserve.ridge.train-only.v1"
    assert len(output["prediction_and_reserve_plan"]) == 3
    assert all("prepared_reserve_kwh" in row for row in output["prediction_and_reserve_plan"])
