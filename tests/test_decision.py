import numpy as np
import pandas as pd
import pytest

from sunreserve.config import ExperimentConfig
from sunreserve.decision import (
    evaluate_decision_scenarios,
    generation_proxy,
    plan_daily_reserve,
    simulate_daily_reserve,
)


def simulate(actual, predicted, *, demand=10.0, capacity=15.0, buffer=0.0):
    return simulate_daily_reserve(
        actual,
        predicted,
        demand_kwh=demand,
        capacity_kwh=capacity,
        buffer_kwh=buffer,
        panel_area_m2=20.0,
        efficiency=0.20,
        derating=0.80,
    )


def test_zero_sun_reserves_full_demand_within_capacity() -> None:
    result = simulate([0.0], [0.0])

    assert result.loc[0, "prepared_reserve_kwh"] == 10.0
    assert result.loc[0, "simulated_shortfall_kwh"] == 0.0
    assert result.loc[0, "excess_prepared_reserve_kwh"] == 0.0


def test_ample_sun_needs_no_reserve_and_has_no_shortfall() -> None:
    result = simulate([10.0], [10.0])

    assert result.loc[0, "actual_generation_proxy_kwh_day"] == 32.0
    assert result.loc[0, "prepared_reserve_kwh"] == 0.0
    assert result.loc[0, "simulated_shortfall_kwh"] == 0.0


def test_exact_energy_balance_has_zero_shortfall_and_excess() -> None:
    result = simulate([3.125], [3.0], demand=10.0)

    assert result.loc[0, "actual_generation_proxy_kwh_day"] == pytest.approx(10.0)
    assert result.loc[0, "prepared_reserve_kwh"] == pytest.approx(0.4)
    assert result.loc[0, "simulated_shortfall_kwh"] == pytest.approx(0.0)
    assert result.loc[0, "excess_prepared_reserve_kwh"] == pytest.approx(0.4)


def test_negative_forecast_is_clipped_to_zero_generation() -> None:
    result = simulate([0.0], [-4.0], demand=10.0)

    assert result.loc[0, "predicted_generation_proxy_kwh_day"] == 0.0
    assert result.loc[0, "prepared_reserve_kwh"] == 10.0


def test_capacity_exhaustion_separates_unavoidable_and_forecast_shortfall() -> None:
    result = simulate([0.0], [0.0], demand=10.0, capacity=5.0)

    assert result.loc[0, "prepared_reserve_kwh"] == 5.0
    assert result.loc[0, "simulated_shortfall_kwh"] == 5.0
    assert result.loc[0, "capacity_limited_shortfall_kwh"] == 5.0
    assert result.loc[0, "additional_shortfall_above_capacity_floor_kwh"] == 0.0


def test_live_reserve_plan_uses_forecast_without_claiming_realized_outcomes() -> None:
    plan = plan_daily_reserve(
        [2.0, 0.0],
        demand_kwh=15.0,
        capacity_kwh=15.0,
        buffer_kwh=0.0,
        panel_area_m2=20.0,
        efficiency=0.20,
        derating=0.80,
    )

    assert plan["predicted_generation_proxy_kwh_day"].tolist() == pytest.approx([6.4, 0.0])
    assert plan["prepared_reserve_kwh"].tolist() == pytest.approx([8.6, 15.0])
    assert "simulated_shortfall_kwh" not in plan
    assert "excess_prepared_reserve_kwh" not in plan


def test_generation_proxy_rejects_overflow_from_finite_inputs() -> None:
    with pytest.raises(ValueError, match="generation proxy calculation produced non-finite"):
        generation_proxy(
            [1e308],
            panel_area_m2=1e308,
            efficiency=0.20,
            derating=0.80,
        )


def test_reserve_plan_rejects_overflow_from_finite_demand_and_buffer() -> None:
    with pytest.raises(ValueError, match="reserve need calculation produced non-finite"):
        plan_daily_reserve(
            [0.0],
            demand_kwh=1.7e308,
            capacity_kwh=1.7e308,
            buffer_kwh=1.7e308,
            panel_area_m2=20.0,
            efficiency=0.20,
            derating=0.80,
        )


def test_simulator_rejects_overflow_in_capacity_limited_calculation() -> None:
    with pytest.raises(ValueError, match="reserve outcome calculation produced non-finite"):
        simulate_daily_reserve(
            [1.7e308],
            [0.0],
            demand_kwh=0.0,
            capacity_kwh=1.7e308,
            buffer_kwh=0.0,
            panel_area_m2=1.0,
            efficiency=1.0,
            derating=1.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"demand_kwh": -1.0},
        {"capacity_kwh": np.nan},
        {"buffer_kwh": np.inf},
    ],
)
def test_simulator_rejects_invalid_scenario_values(kwargs) -> None:
    values = {"demand_kwh": 10.0, "capacity_kwh": 15.0, "buffer_kwh": 0.0}
    values.update(kwargs)
    with pytest.raises(ValueError):
        simulate_daily_reserve(
            [0.0],
            [0.0],
            **values,
            panel_area_m2=20.0,
            efficiency=0.20,
            derating=0.80,
        )


def test_grid_uses_identical_dates_settings_and_observations_across_approaches() -> None:
    dates = pd.date_range("2022-01-01", periods=3, freq="D")
    predictions = pd.DataFrame(
        {
            "target_date": dates,
            "target_radiation_kwh_m2_day": [0.0, 2.0, 8.0],
            "persistence_prediction_kwh_m2_day": [0.0, 1.0, 7.0],
            "seasonal_mean_prediction_kwh_m2_day": [1.0, 2.0, 3.0],
            "ridge_prediction_kwh_m2_day": [0.5, 2.5, 8.0],
        }
    )

    results = evaluate_decision_scenarios(predictions, ExperimentConfig(), selected_model="ridge")

    assert results.metrics["scenario_id"].nunique() == 28  # primary plus fixed 27-case grid
    assert len(results.metrics) == 28 * 4
    assert len(results.predictions) == 28 * 4 * len(dates)
    for _, group in results.predictions.groupby("scenario_id", sort=False):
        assert set(group["approach"]) == {
            "persistence",
            "seasonal_mean",
            "ridge",
            "always_prepare_backup",
        }
        assert group.groupby("approach")["target_date"].apply(tuple).nunique() == 1
        for column in (
            "actual_radiation_kwh_m2_day",
            "demand_kwh",
            "capacity_kwh",
            "buffer_kwh",
            "actual_generation_proxy_kwh_day",
            "capacity_limited_shortfall_kwh",
        ):
            assert group.groupby("approach")[column].apply(tuple).nunique() == 1
        backup = group.loc[group["approach"] == "always_prepare_backup"]
        assert backup["simulated_shortfall_kwh"].tolist() == pytest.approx(
            backup["capacity_limited_shortfall_kwh"].tolist()
        )
        assert (group["additional_shortfall_above_capacity_floor_kwh"] >= 0).all()
    assert (
        results.metrics["paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day"]
        .notna()
        .all()
    )
    assert (
        results.metrics["paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day"]
        .notna()
        .all()
    )
    assert results.metrics["paired_mean_excess_difference_vs_persistence_kwh_day"].notna().all()
    assert results.metrics["paired_mean_excess_difference_vs_seasonal_mean_kwh_day"].notna().all()
