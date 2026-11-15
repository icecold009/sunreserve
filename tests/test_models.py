import pandas as pd
import pytest

from sunreserve.config import ExperimentConfig
from sunreserve.evaluate import SELECTION_TIE_ORDER, select_validation_model
from sunreserve.features import FEATURE_COLUMNS, TARGET_COLUMN
from sunreserve.train import (
    baseline_predictions,
    fit_ridge,
    fit_seasonal_mean,
    predict_persistence,
    predict_ridge,
)


def training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "train", "train"],
            "issue_date": pd.to_datetime(["2020-01-01", "2020-01-31", "2020-02-01"]),
            "target_date": pd.to_datetime(["2020-01-02", "2020-02-01", "2020-02-02"]),
            TARGET_COLUMN: [10.0, 30.0, 20.0],
            "radiation_today_kwh_m2_day": [8.0, 10.0, 15.0],
        }
    )


def test_seasonal_baseline_uses_target_month_means_from_training_only() -> None:
    train = training_rows()
    model = fit_seasonal_mean(train)
    evaluation = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "target_date": pd.to_datetime(["2020-01-15", "2020-03-01"]),
            TARGET_COLUMN: [9999.0, 9999.0],
            "radiation_today_kwh_m2_day": [5.0, 6.0],
            "issue_date": pd.to_datetime(["2020-01-14", "2020-02-29"]),
        }
    )

    predictions = model.predict(evaluation)

    assert model.monthly_means == {1: 10.0, 2: 25.0}
    assert model.training_global_mean == pytest.approx(20.0)
    assert model.training_rows == 3
    assert predictions.tolist() == pytest.approx([10.0, 20.0])


def test_seasonal_baseline_refuses_validation_or_test_labels() -> None:
    mixed = training_rows()
    mixed.loc[0, "split"] = "validation"

    with pytest.raises(ValueError, match="training rows only"):
        fit_seasonal_mean(mixed)


def test_persistence_clips_negative_prediction_and_preserves_row_alignment() -> None:
    rows = pd.DataFrame(
        {"radiation_today_kwh_m2_day": [-0.2, 4.5]},
        index=pd.Index([9, 4], name="row"),
    )

    predictions = predict_persistence(rows)

    assert predictions.index.tolist() == [9, 4]
    assert predictions.tolist() == pytest.approx([0.0, 4.5])


def test_both_baselines_predict_the_same_eligible_validation_dates() -> None:
    train = training_rows()
    evaluation = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "issue_date": pd.to_datetime(["2022-02-27", "2022-02-28"]),
            "target_date": pd.to_datetime(["2022-02-28", "2022-03-01"]),
            TARGET_COLUMN: [2.0, 3.0],
            "radiation_today_kwh_m2_day": [1.0, 2.5],
        }
    )

    result = baseline_predictions(train, evaluation)

    assert result["target_date"].tolist() == evaluation["target_date"].tolist()
    assert result["persistence_prediction_kwh_m2_day"].tolist() == pytest.approx([1.0, 2.5])
    assert result["seasonal_mean_prediction_kwh_m2_day"].tolist() == pytest.approx([25.0, 20.0])


def test_final_test_predictions_require_explicit_finalization() -> None:
    evaluation = pd.DataFrame(
        {
            "split": ["test"],
            "issue_date": pd.to_datetime(["2023-01-01"]),
            "target_date": pd.to_datetime(["2023-01-02"]),
            TARGET_COLUMN: [1.5],
            "radiation_today_kwh_m2_day": [1.0],
        }
    )

    with pytest.raises(ValueError, match="explicit finalization"):
        baseline_predictions(training_rows(), evaluation)


def ridge_rows(split: str, targets: list[float]) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "split": [split] * len(targets),
            "issue_date": pd.date_range("2020-01-01", periods=len(targets)),
            "target_date": pd.date_range("2020-01-02", periods=len(targets)),
            TARGET_COLUMN: targets,
            "radiation_today_kwh_m2_day": targets,
        }
    )
    for index, feature in enumerate(FEATURE_COLUMNS):
        rows[feature] = [float(row + index) for row in range(len(targets))]
    return rows


def test_ridge_imputer_and_scaler_statistics_come_from_training_rows_only() -> None:
    train = ridge_rows("train", [1.0, 2.0, 3.0])
    train[FEATURE_COLUMNS[0]] = [1.0, float("nan"), 5.0]
    model = fit_ridge(train, ExperimentConfig())
    validation = ridge_rows("validation", [1.0, 2.0])
    validation[FEATURE_COLUMNS[0]] = [10000.0, float("nan")]
    imputer_before = model.named_steps["imputer"].statistics_.copy()
    scaler_before = model.named_steps["scaler"].mean_.copy()

    predictions = predict_ridge(model, validation)

    assert model.named_steps["imputer"].statistics_[0] == pytest.approx(3.0)
    assert model.named_steps["scaler"].mean_[0] == pytest.approx(3.0)
    assert model.named_steps["imputer"].statistics_.tolist() == imputer_before.tolist()
    assert model.named_steps["scaler"].mean_.tolist() == scaler_before.tolist()
    assert len(predictions) == len(validation)
    assert (predictions >= 0).all()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (pd.DataFrame(), "training rows must contain"),
        (ridge_rows("validation", [1.0]), "training rows only"),
    ],
)
def test_ridge_fit_rejects_empty_or_nontraining_partitions(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_ridge(rows, ExperimentConfig())


def test_ridge_fit_rejects_infinite_and_entirely_missing_training_features() -> None:
    train = ridge_rows("train", [1.0, 2.0, 3.0])
    train.loc[0, FEATURE_COLUMNS[0]] = float("inf")
    with pytest.raises(ValueError, match="infinite"):
        fit_ridge(train, ExperimentConfig())

    train = ridge_rows("train", [1.0, 2.0, 3.0])
    train[FEATURE_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError, match="entirely missing"):
        fit_ridge(train, ExperimentConfig())


def test_validation_selection_retains_baselines_and_uses_fixed_tie_order() -> None:
    train = ridge_rows("train", [0.0, 0.0, 0.0])
    validation = ridge_rows("validation", [0.0, 0.0])
    validation["target_date"] = pd.date_range("2022-01-01", periods=len(validation))
    validation["radiation_today_kwh_m2_day"] = 0.0
    result = select_validation_model(train, validation, ExperimentConfig())

    assert set(result.metrics) == set(SELECTION_TIE_ORDER)
    assert result.selected_model == "persistence"
    assert all(metrics["mae"] == 0 for metrics in result.metrics.values())
    assert result.low_solar_threshold == 0.0
    assert result.predictions["target_date"].tolist() == validation["target_date"].tolist()
    assert result.eligible_rows == {"train": 3, "validation": 2}


def test_validation_selection_refuses_test_rows_and_out_of_range_validation() -> None:
    train = ridge_rows("train", [1.0, 2.0, 3.0])
    test = ridge_rows("test", [1000.0, 2000.0])
    with pytest.raises(ValueError, match="validation rows only"):
        select_validation_model(train, test, ExperimentConfig())

    validation = ridge_rows("validation", [1.0, 2.0])
    validation["target_date"] = pd.to_datetime(["2023-01-01", "2023-01-02"])
    with pytest.raises(ValueError, match="outside configured validation dates"):
        select_validation_model(train, validation, ExperimentConfig())


def test_low_solar_threshold_is_computed_from_training_targets_only() -> None:
    train = ridge_rows("train", [10.0, 20.0, 30.0, 40.0])
    validation = ridge_rows("validation", [10_000.0, 20_000.0])
    validation["target_date"] = pd.date_range("2022-01-01", periods=len(validation))

    selection = select_validation_model(train, validation, ExperimentConfig())

    assert selection.low_solar_threshold == pytest.approx(17.5)
