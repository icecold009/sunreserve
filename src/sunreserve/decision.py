"""Transparent daily reserve-energy scenarios for historical research replay."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from sunreserve.config import ExperimentConfig

RADIATION_COLUMN = "target_radiation_kwh_m2_day"
PREDICTION_COLUMNS = {
    "persistence": "persistence_prediction_kwh_m2_day",
    "seasonal_mean": "seasonal_mean_prediction_kwh_m2_day",
    "ridge": "ridge_prediction_kwh_m2_day",
}
APPROACHES = (*PREDICTION_COLUMNS, "always_prepare_backup")
DEMAND_LEVELS_KWH = (10.0, 15.0, 20.0)
CAPACITY_LEVELS_KWH = (10.0, 15.0, 20.0)
BUFFER_FRACTIONS = (0.0, 0.1, 0.2)


@dataclass(frozen=True)
class DecisionResults:
    """Per-date scenario decisions and paired validation summaries."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _finite_vector(values: Any, name: str, *, nonnegative: bool = False) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite numeric values") from error
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and (vector < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return vector


def _finite_nonnegative_scalar(value: float, name: str) -> float:
    array = _finite_vector([value], name, nonnegative=True)
    return float(array[0])


def _require_finite_output(values: Any, name: str) -> None:
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError(f"{name} calculation produced non-finite values; reduce input magnitudes")


def generation_proxy(
    radiation_kwh_m2_day: Any,
    *,
    panel_area_m2: float,
    efficiency: float,
    derating: float,
) -> np.ndarray:
    """Convert daily solar radiation to a hypothetical electricity proxy in kWh/day."""
    radiation = _finite_vector(radiation_kwh_m2_day, "radiation", nonnegative=True)
    area = _finite_nonnegative_scalar(panel_area_m2, "panel_area_m2")
    efficiency_value = _finite_nonnegative_scalar(efficiency, "efficiency")
    derating_value = _finite_nonnegative_scalar(derating, "derating")
    if area <= 0 or not 0 < efficiency_value <= 1 or not 0 < derating_value <= 1:
        raise ValueError("area must be positive and efficiency/derating must be in (0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        generation = radiation * area * efficiency_value * derating_value
    _require_finite_output(generation, "generation proxy")
    return generation


def plan_daily_reserve(
    predicted_radiation_kwh_m2_day: Any,
    *,
    demand_kwh: float,
    capacity_kwh: float,
    buffer_kwh: float,
    panel_area_m2: float,
    efficiency: float,
    derating: float,
) -> pd.DataFrame:
    """Plan reserve from forecast values without inventing an observed outcome.

    This function intentionally does not calculate realized shortfall or excess reserve;
    those require the next day's actual solar radiation.
    """
    predicted = np.maximum(
        _finite_vector(predicted_radiation_kwh_m2_day, "predicted_radiation"), 0.0
    )
    if len(predicted) == 0:
        raise ValueError("predicted radiation must contain at least one value")
    demand = _finite_nonnegative_scalar(demand_kwh, "demand_kwh")
    capacity = _finite_nonnegative_scalar(capacity_kwh, "capacity_kwh")
    buffer = _finite_nonnegative_scalar(buffer_kwh, "buffer_kwh")
    predicted_generation = generation_proxy(
        predicted,
        panel_area_m2=panel_area_m2,
        efficiency=efficiency,
        derating=derating,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        reserve_need = demand - predicted_generation + buffer
    _require_finite_output(reserve_need, "reserve need")
    reserve = np.minimum(capacity, np.maximum(0.0, reserve_need))
    _require_finite_output(reserve, "planned reserve")
    return pd.DataFrame(
        {
            "predicted_radiation_kwh_m2_day": predicted,
            "predicted_generation_proxy_kwh_day": predicted_generation,
            "demand_kwh": demand,
            "capacity_kwh": capacity,
            "buffer_kwh": buffer,
            "prepared_reserve_kwh": reserve,
        }
    )


def simulate_daily_reserve(
    actual_radiation_kwh_m2_day: Any,
    predicted_radiation_kwh_m2_day: Any,
    *,
    demand_kwh: float,
    capacity_kwh: float,
    buffer_kwh: float,
    panel_area_m2: float,
    efficiency: float,
    derating: float,
) -> pd.DataFrame:
    """Calculate daily reserve, forecast-dependent shortfall, and excess reserve."""
    actual_radiation = _finite_vector(
        actual_radiation_kwh_m2_day, "actual_radiation", nonnegative=True
    )
    predicted_radiation = np.maximum(
        _finite_vector(predicted_radiation_kwh_m2_day, "predicted_radiation"), 0.0
    )
    if len(actual_radiation) == 0 or len(actual_radiation) != len(predicted_radiation):
        raise ValueError("actual and predicted radiation must have equal nonzero lengths")
    demand = _finite_nonnegative_scalar(demand_kwh, "demand_kwh")
    capacity = _finite_nonnegative_scalar(capacity_kwh, "capacity_kwh")
    buffer = _finite_nonnegative_scalar(buffer_kwh, "buffer_kwh")

    actual_generation = generation_proxy(
        actual_radiation,
        panel_area_m2=panel_area_m2,
        efficiency=efficiency,
        derating=derating,
    )
    predicted_generation = generation_proxy(
        predicted_radiation,
        panel_area_m2=panel_area_m2,
        efficiency=efficiency,
        derating=derating,
    )
    reserve = plan_daily_reserve(
        predicted_radiation,
        demand_kwh=demand,
        capacity_kwh=capacity,
        buffer_kwh=buffer,
        panel_area_m2=panel_area_m2,
        efficiency=efficiency,
        derating=derating,
    )["prepared_reserve_kwh"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        net_demand = demand - actual_generation
        shortfall_need = net_demand - reserve
        excess_need = reserve - np.maximum(0.0, net_demand)
        capacity_limited_need = net_demand - capacity
    _require_finite_output(
        np.concatenate((shortfall_need, excess_need, capacity_limited_need)),
        "reserve outcome",
    )
    shortfall = np.maximum(0.0, shortfall_need)
    excess_reserve = np.maximum(0.0, excess_need)
    capacity_limited = np.maximum(0.0, capacity_limited_need)

    return pd.DataFrame(
        {
            "actual_generation_proxy_kwh_day": actual_generation,
            "predicted_generation_proxy_kwh_day": predicted_generation,
            "demand_kwh": demand,
            "capacity_kwh": capacity,
            "buffer_kwh": buffer,
            "prepared_reserve_kwh": reserve,
            "simulated_shortfall_kwh": shortfall,
            "excess_prepared_reserve_kwh": excess_reserve,
            "capacity_limited_shortfall_kwh": capacity_limited,
            "additional_shortfall_above_capacity_floor_kwh": np.maximum(
                0.0, shortfall - capacity_limited
            ),
        }
    )


def _summarize(
    frame: pd.DataFrame,
    *,
    selected_model: str,
    scenario_id: str,
) -> dict[str, Any]:
    shortfall = frame["simulated_shortfall_kwh"].to_numpy(dtype=float)
    excess = frame["excess_prepared_reserve_kwh"].to_numpy(dtype=float)
    capacity_limited = frame["capacity_limited_shortfall_kwh"].to_numpy(dtype=float)
    return {
        "scenario_id": scenario_id,
        "approach": str(frame["approach"].iloc[0]),
        "selected_model": selected_model,
        "sample_count": int(len(frame)),
        "simulated_shortfall_day_count": int(np.count_nonzero(shortfall > 0)),
        "simulated_shortfall_day_share": float(np.mean(shortfall > 0)),
        "total_simulated_shortfall_kwh": float(shortfall.sum()),
        "mean_simulated_shortfall_kwh_day": float(shortfall.mean()),
        "total_excess_prepared_reserve_kwh": float(excess.sum()),
        "mean_excess_prepared_reserve_kwh_day": float(excess.mean()),
        "capacity_limited_shortfall_day_count": int(np.count_nonzero(capacity_limited > 0)),
        "total_capacity_limited_shortfall_kwh": float(capacity_limited.sum()),
        "mean_capacity_limited_shortfall_kwh_day": float(capacity_limited.mean()),
        "total_additional_shortfall_above_capacity_floor_kwh": float(
            frame["additional_shortfall_above_capacity_floor_kwh"].sum()
        ),
        "mean_additional_shortfall_above_capacity_floor_kwh_day": float(
            frame["additional_shortfall_above_capacity_floor_kwh"].mean()
        ),
        "paired_mean_simulated_shortfall_difference_vs_persistence_kwh_day": None,
        "paired_mean_simulated_shortfall_difference_vs_seasonal_mean_kwh_day": None,
        "paired_mean_excess_difference_vs_persistence_kwh_day": None,
        "paired_mean_excess_difference_vs_seasonal_mean_kwh_day": None,
    }


def evaluate_decision_scenarios(
    validation_predictions: pd.DataFrame,
    config: ExperimentConfig,
    *,
    selected_model: str,
) -> DecisionResults:
    """Evaluate the fixed reserve grid on one common set of validation target dates."""
    if selected_model not in PREDICTION_COLUMNS:
        raise ValueError(f"unsupported selected model {selected_model!r}")
    required = {"target_date", RADIATION_COLUMN, *PREDICTION_COLUMNS.values()}
    missing = required - set(validation_predictions.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"validation predictions are missing columns: {names}")
    dates = pd.DatetimeIndex(pd.to_datetime(validation_predictions["target_date"]))
    if len(dates) == 0 or dates.has_duplicates or dates.isna().any():
        raise ValueError("validation target dates must be nonempty, valid, and unique")
    actual_radiation = _finite_vector(
        validation_predictions[RADIATION_COLUMN], "actual_radiation", nonnegative=True
    )
    clipped_predictions = {
        name: np.maximum(_finite_vector(validation_predictions[column], name), 0.0)
        for name, column in PREDICTION_COLUMNS.items()
    }
    if len(actual_radiation) != len(dates):
        raise ValueError("validation dates and observed radiation must have equal lengths")

    settings = [
        (
            "primary",
            config.daily_demand_kwh,
            config.reserve_cap_kwh,
            config.reserve_buffer_kwh,
        ),
        *(
            (
                f"grid_d{demand:g}_c{capacity:g}_b{fraction:g}",
                demand,
                capacity,
                demand * fraction,
            )
            for demand, capacity, fraction in product(
                DEMAND_LEVELS_KWH, CAPACITY_LEVELS_KWH, BUFFER_FRACTIONS
            )
        ),
    ]
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for scenario_id, demand, capacity, buffer in settings:
        scenario_frames: dict[str, pd.DataFrame] = {}
        for approach, predicted in clipped_predictions.items():
            metrics = simulate_daily_reserve(
                actual_radiation,
                predicted,
                demand_kwh=demand,
                capacity_kwh=capacity,
                buffer_kwh=buffer,
                panel_area_m2=config.panel_area_m2,
                efficiency=config.conversion_efficiency,
                derating=config.derating_factor,
            )
            metrics.insert(0, "target_date", dates)
            metrics.insert(0, "approach", approach)
            metrics.insert(0, "scenario_id", scenario_id)
            metrics.insert(3, "actual_radiation_kwh_m2_day", actual_radiation)
            metrics.insert(4, "predicted_radiation_kwh_m2_day", predicted)
            scenario_frames[approach] = metrics

        backup_reserve = min(demand, capacity)
        backup = pd.DataFrame(
            {
                "actual_generation_proxy_kwh_day": generation_proxy(
                    actual_radiation,
                    panel_area_m2=config.panel_area_m2,
                    efficiency=config.conversion_efficiency,
                    derating=config.derating_factor,
                ),
                "predicted_generation_proxy_kwh_day": np.nan,
                "demand_kwh": demand,
                "capacity_kwh": capacity,
                "buffer_kwh": buffer,
                "prepared_reserve_kwh": backup_reserve,
                "simulated_shortfall_kwh": np.maximum(
                    0.0,
                    demand
                    - generation_proxy(
                        actual_radiation,
                        panel_area_m2=config.panel_area_m2,
                        efficiency=config.conversion_efficiency,
                        derating=config.derating_factor,
                    )
                    - backup_reserve,
                ),
                "excess_prepared_reserve_kwh": np.maximum(
                    0.0,
                    backup_reserve
                    - np.maximum(
                        0.0,
                        demand
                        - generation_proxy(
                            actual_radiation,
                            panel_area_m2=config.panel_area_m2,
                            efficiency=config.conversion_efficiency,
                            derating=config.derating_factor,
                        ),
                    ),
                ),
                "capacity_limited_shortfall_kwh": np.maximum(
                    0.0,
                    demand
                    - generation_proxy(
                        actual_radiation,
                        panel_area_m2=config.panel_area_m2,
                        efficiency=config.conversion_efficiency,
                        derating=config.derating_factor,
                    )
                    - capacity,
                ),
            }
        )
        backup["additional_shortfall_above_capacity_floor_kwh"] = np.maximum(
            0.0,
            backup["simulated_shortfall_kwh"] - backup["capacity_limited_shortfall_kwh"],
        )
        backup.insert(0, "target_date", dates)
        backup.insert(0, "approach", "always_prepare_backup")
        backup.insert(0, "scenario_id", scenario_id)
        backup.insert(3, "actual_radiation_kwh_m2_day", actual_radiation)
        backup.insert(4, "predicted_radiation_kwh_m2_day", np.nan)
        scenario_frames["always_prepare_backup"] = backup

        for approach in APPROACHES:
            approach_frame = scenario_frames[approach]
            prediction_frames.append(approach_frame)
            metric_rows.append(
                _summarize(approach_frame, selected_model=selected_model, scenario_id=scenario_id)
            )

        means = {
            approach: {
                "shortfall": float(frame["simulated_shortfall_kwh"].mean()),
                "excess": float(frame["excess_prepared_reserve_kwh"].mean()),
            }
            for approach, frame in scenario_frames.items()
        }
        for row in metric_rows[-len(APPROACHES) :]:
            approach = row["approach"]
            for baseline in ("persistence", "seasonal_mean"):
                row[f"paired_mean_simulated_shortfall_difference_vs_{baseline}_kwh_day"] = (
                    means[approach]["shortfall"] - means[baseline]["shortfall"]
                )
                row[f"paired_mean_excess_difference_vs_{baseline}_kwh_day"] = (
                    means[approach]["excess"] - means[baseline]["excess"]
                )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_metrics = pd.DataFrame(metric_rows)
    return DecisionResults(predictions=all_predictions, metrics=all_metrics)
