import json
import math
from datetime import date

import pytest

from sunreserve.config import ExperimentConfig


def test_configuration_round_trip_is_json_serializable(tmp_path) -> None:
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    loaded = ExperimentConfig.load(path)

    assert loaded == config
    assert loaded.parameters == ("ALLSKY_SFC_SW_DWN", "T2M", "RH2M")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"latitude": 91}, "latitude"),
        ({"longitude": 181}, "longitude"),
        ({"start_date": date(2025, 1, 1), "end_date": date(2024, 1, 1)}, "start_date"),
        ({"conversion_efficiency": 0}, "conversion_efficiency"),
        ({"derating_factor": 1.1}, "derating_factor"),
        ({"panel_area_m2": 0}, "panel_area_m2"),
        ({"daily_demand_kwh": 0}, "daily_demand_kwh"),
        ({"reserve_cap_kwh": -1}, "reserve capacity"),
        ({"reserve_buffer_kwh": -1}, "reserve capacity"),
        ({"ridge_alpha": 0}, "ridge_alpha"),
        ({"low_solar_quantile": 1}, "low_solar_quantile"),
        ({"primary_metric": "rmse"}, "primary_metric"),
        ({"training_end": date(2022, 1, 1)}, "boundaries must be disjoint"),
    ],
)
def test_configuration_rejects_invalid_values(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**values)


def test_configuration_rejects_unknown_json_keys() -> None:
    with pytest.raises(ValueError, match="unknown configuration keys"):
        ExperimentConfig.from_dict({"mystery": True})


@pytest.mark.parametrize(
    "allowed_models",
    [
        ("persistence", "ridge"),
        ("persistence", "seasonal_mean", "ridge", "random_forest"),
        ("persistence", "seasonal_mean", "seasonal_mean", "ridge"),
    ],
)
def test_configuration_requires_each_supported_comparison_model_once(allowed_models) -> None:
    with pytest.raises(ValueError, match="allowed_models must contain exactly"):
        ExperimentConfig(allowed_models=allowed_models)


def test_configuration_from_dict_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="allowed_models must contain exactly"):
        ExperimentConfig.from_dict(
            {"allowed_models": ["persistence", "seasonal_mean", "neural_net"]}
        )


@pytest.mark.parametrize(
    "field",
    [
        "latitude",
        "longitude",
        "panel_area_m2",
        "conversion_efficiency",
        "derating_factor",
        "daily_demand_kwh",
        "reserve_cap_kwh",
        "reserve_buffer_kwh",
        "ridge_alpha",
        "low_solar_quantile",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_configuration_rejects_nonfinite_numeric_values(field, value) -> None:
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        ExperimentConfig(**{field: value})
