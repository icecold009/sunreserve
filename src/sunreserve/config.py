"""Validated, serializable settings for the SunReserve experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment inputs and declared assumptions; values are not measured facility data."""

    REQUIRED_MODELS = frozenset({"persistence", "seasonal_mean", "ridge"})

    latitude: float = 12.9716
    longitude: float = 77.5946
    start_date: date = date(2016, 1, 1)
    end_date: date = date(2024, 12, 31)
    time_standard: str = "UTC"
    community: str = "RE"
    parameters: tuple[str, ...] = ("ALLSKY_SFC_SW_DWN", "T2M", "RH2M")
    training_end: date = date(2021, 12, 31)
    validation_start: date = date(2022, 1, 1)
    validation_end: date = date(2022, 12, 31)
    test_start: date = date(2023, 1, 1)
    random_seed: int = 42
    allowed_models: tuple[str, ...] = ("persistence", "seasonal_mean", "ridge")
    primary_metric: str = "mae"
    ridge_alpha: float = 1.0
    low_solar_quantile: float = 0.25
    panel_area_m2: float = 20.0
    conversion_efficiency: float = 0.20
    derating_factor: float = 0.80
    daily_demand_kwh: float = 15.0
    reserve_cap_kwh: float = 15.0
    reserve_buffer_kwh: float = 0.0
    source: str = "NASA POWER Daily API"

    def __post_init__(self) -> None:
        numeric_values = (
            ("latitude", self.latitude),
            ("longitude", self.longitude),
            ("panel_area_m2", self.panel_area_m2),
            ("conversion_efficiency", self.conversion_efficiency),
            ("derating_factor", self.derating_factor),
            ("daily_demand_kwh", self.daily_demand_kwh),
            ("reserve_cap_kwh", self.reserve_cap_kwh),
            ("reserve_buffer_kwh", self.reserve_buffer_kwh),
            ("ridge_alpha", self.ridge_alpha),
            ("low_solar_quantile", self.low_solar_quantile),
        )
        for name, value in numeric_values:
            try:
                finite = math.isfinite(value)
            except TypeError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be finite")
        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180 degrees")
        if self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        if self.time_standard not in {"UTC", "LST"}:
            raise ValueError("time_standard must be UTC or LST")
        if self.primary_metric != "mae":
            raise ValueError("primary_metric must be mae for validation selection")
        if (
            len(set(self.allowed_models)) != len(self.allowed_models)
            or set(self.allowed_models) != self.REQUIRED_MODELS
        ):
            required = ", ".join(sorted(self.REQUIRED_MODELS))
            raise ValueError(f"allowed_models must contain exactly these models once: {required}")
        if not self.community:
            raise ValueError("community must be non-empty")
        if len(set(self.parameters)) != len(self.parameters) or not self.parameters:
            raise ValueError("parameters must be a non-empty list without duplicates")
        if not (self.training_end < self.validation_start <= self.validation_end < self.test_start):
            raise ValueError("training, validation, and test date boundaries must be disjoint")
        for name, value in (
            ("panel_area_m2", self.panel_area_m2),
            ("daily_demand_kwh", self.daily_demand_kwh),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("conversion_efficiency", self.conversion_efficiency),
            ("derating_factor", self.derating_factor),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.reserve_cap_kwh < 0 or self.reserve_buffer_kwh < 0:
            raise ValueError("reserve capacity and buffer must be nonnegative")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if not 0 < self.low_solar_quantile < 1:
            raise ValueError("low_solar_quantile must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready values with ISO-8601 dates."""
        return {
            **asdict(self),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "training_end": self.training_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "parameters": list(self.parameters),
            "allowed_models": list(self.allowed_models),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExperimentConfig:
        """Build a config from JSON-compatible values, rejecting unknown keys."""
        known = {name for name in cls.__dataclass_fields__ if not name.startswith("_")}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
        converted = dict(values)
        for name in (
            "start_date",
            "end_date",
            "training_end",
            "validation_start",
            "validation_end",
            "test_start",
        ):
            if name in converted and isinstance(converted[name], str):
                converted[name] = date.fromisoformat(converted[name])
        for name in ("parameters", "allowed_models"):
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read configuration {path}: {error}") from error
        if not isinstance(values, dict):
            raise ValueError("configuration JSON must contain an object")
        return cls.from_dict(values)
