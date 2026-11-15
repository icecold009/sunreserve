import numpy as np
import pandas as pd
import pytest

from sunreserve.evaluate import clip_predictions, forecast_metrics


def test_forecast_metrics_match_hand_calculated_mae_rmse_bias_and_groups() -> None:
    actual = [1.0, 2.0, 3.0, 4.0]
    predictions = [-2.0, 2.0, 5.0, 3.0]
    dates = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-04-15", "2022-08-01"])

    metrics = forecast_metrics(
        actual,
        predictions,
        dates,
        low_solar_threshold=2.5,
    )

    assert metrics["sample_count"] == 4
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.5))
    assert metrics["bias_prediction_minus_actual"] == pytest.approx(0.0)
    assert metrics["monthly_mae_by_target_calendar_month"] == {
        "01": pytest.approx(0.5),
        "04": pytest.approx(2.0),
        "08": pytest.approx(1.0),
    }
    assert metrics["seasonal_mae"] == {
        "DJF": pytest.approx(0.5),
        "MAM": pytest.approx(2.0),
        "JJA": pytest.approx(1.0),
    }
    assert metrics["low_solar"] == {
        "threshold_kwh_m2_day": 2.5,
        "sample_count": 2,
        "mae": pytest.approx(0.5),
        "rmse": pytest.approx(np.sqrt(0.5)),
        "bias_prediction_minus_actual": pytest.approx(-0.5),
    }
    assert "max(0, prediction)" in metrics["prediction_clip_rule"]


def test_low_solar_metrics_are_null_when_no_validation_day_is_below_train_threshold() -> None:
    metrics = forecast_metrics(
        [3.0, 4.0],
        [2.0, 5.0],
        pd.to_datetime(["2022-01-01", "2022-01-02"]),
        low_solar_threshold=1.0,
    )

    assert metrics["low_solar"] == {
        "threshold_kwh_m2_day": 1.0,
        "sample_count": 0,
        "mae": None,
        "rmse": None,
        "bias_prediction_minus_actual": None,
    }


@pytest.mark.parametrize(
    ("actual", "prediction", "dates", "threshold", "message"),
    [
        ([], [], [], 1.0, "empty partition"),
        ([1.0], [1.0, 2.0], ["2022-01-01"], 1.0, "equal lengths"),
        ([1.0], [float("nan")], ["2022-01-01"], 1.0, "predictions must contain"),
        ([float("inf")], [1.0], ["2022-01-01"], 1.0, "actual must contain"),
        ([-1.0], [1.0], ["2022-01-01"], 1.0, "actual solar radiation"),
        ([1.0], [1.0], ["2022-01-01"], -1.0, "threshold"),
        ([1.0, 2.0], [1.0, 2.0], ["2022-01-01", "2022-01-01"], 1.0, "unique"),
    ],
)
def test_metrics_reject_empty_invalid_or_misaligned_inputs(
    actual, prediction, dates, threshold, message
) -> None:
    with pytest.raises(ValueError, match=message):
        forecast_metrics(actual, prediction, dates, low_solar_threshold=threshold)


def test_clipping_handles_only_finite_forecasts() -> None:
    assert clip_predictions([-2.0, 1.5]).tolist() == [0.0, 1.5]
    with pytest.raises(ValueError, match="finite"):
        clip_predictions([float("inf")])
