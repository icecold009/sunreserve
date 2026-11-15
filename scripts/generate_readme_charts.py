#!/usr/bin/env python3
"""Render README charts from the committed frozen 2025 CSV results."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "frozen-test-2025"
ASSETS = ROOT / "docs" / "assets"
WIDTH, HEIGHT = 1200, 650
NAVY = "#15324b"
YELLOW = "#e1a72b"
TEAL = "#168b88"
PURPLE = "#8e6bb8"
OFF_WHITE = "#f7f5ee"
GRID = "#d9e0dc"
MUTED = "#66808c"
SERIES = {
    "Observed": NAVY,
    "Ridge": TEAL,
    "Persistence": YELLOW,
    "Seasonal mean": PURPLE,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_target_dates(rows: list[dict[str, str]]) -> list[date]:
    dates = [date.fromisoformat(row["target_date"]) for row in rows]
    if (
        len(dates) != 365
        or dates[0].isoformat() != "2025-01-01"
        or dates[-1].isoformat() != "2025-12-31"
    ):
        raise ValueError("expected 365 consecutive 2025 targets")
    if any((right - left).days != 1 for left, right in zip(dates, dates[1:], strict=False)):
        raise ValueError("prediction target dates must be consecutive")
    return dates


def radiation_mae_values(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot calculate MAE for an empty prediction table")
    actual = [float(row["target_radiation_kwh_m2_day"]) for row in rows]
    fields = {
        "Ridge": "ridge_prediction_kwh_m2_day",
        "Persistence": "persistence_prediction_kwh_m2_day",
        "Seasonal mean": "seasonal_mean_prediction_kwh_m2_day",
    }
    return {
        name: sum(abs(float(row[field]) - truth) for row, truth in zip(rows, actual, strict=True))
        / len(rows)
        for name, field in fields.items()
    }


def color(value: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(value)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), color(OFF_WHITE))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (16, 16, WIDTH - 16, HEIGHT - 16), radius=26, outline=color("#e7e5dc"), width=2
    )
    draw.text((64, 48), title, fill=color(NAVY), font=font(34))
    draw.text((66, 98), subtitle, fill=color(MUTED), font=font(20))
    return image, draw


def grid(
    draw: ImageDraw.ImageDraw,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
) -> None:
    for value, label in y_ticks:
        y = bottom - (bottom - top) * value
        draw.line((left, y, right, y), fill=color(GRID), width=2)
        draw.text((left - 18, y), label, fill=color(MUTED), font=font(17), anchor="rm")
    for value, label in x_ticks:
        x = left + (right - left) * value
        draw.line((x, top, x, bottom), fill=color(GRID), width=2)
        draw.text((x, bottom + 18), label, fill=color(MUTED), font=font(17), anchor="mt")


def january_chart(rows: list[dict[str, str]]) -> Image.Image:
    jan = [row for row in rows if row["target_date"].startswith("2025-01-")]
    if (
        len(jan) != 31
        or jan[0]["target_date"] != "2025-01-01"
        or jan[-1]["target_date"] != "2025-01-31"
    ):
        raise ValueError("expected 31 consecutive January 2025 target dates")
    left, top, right, bottom = 118, 162, 1110, 526
    maximum = 10.0
    image, draw = canvas(
        "January 2025 · observed vs next-day estimates",
        "Solar radiation · kWh per square metre per day · 31 targets from frozen 2025 results",
    )
    ticks = [((day - 1) / 30, f"{day:02d} Jan") for day in (1, 5, 10, 15, 20, 25, 31)]
    grid(
        draw,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        x_ticks=ticks,
        y_ticks=[(v / maximum, str(v)) for v in range(0, 11, 2)],
    )
    lines = {
        "Observed": [float(row["target_radiation_kwh_m2_day"]) for row in jan],
        "Ridge": [float(row["ridge_prediction_kwh_m2_day"]) for row in jan],
        "Persistence": [float(row["persistence_prediction_kwh_m2_day"]) for row in jan],
    }
    for name, series in lines.items():
        points = [
            (
                round(left + (right - left) * i / 30),
                round(bottom - (bottom - top) * value / maximum),
            )
            for i, value in enumerate(series)
        ]
        draw.line(points, fill=color(SERIES[name]), width=5, joint="curve")
    for i, name in enumerate(lines):
        x = 410 + i * 250
        draw.line((x, 594, x + 48, 594), fill=color(SERIES[name]), width=5)
        draw.text((x + 60, 594), name, fill=color(NAVY), font=font(18), anchor="lm")
    return image


def mae_chart(rows: list[dict[str, str]]) -> Image.Image:
    names = ["Ridge", "Persistence", "Seasonal mean"]
    maes = radiation_mae_values(rows)
    left, top, right, bottom = 130, 164, 1094, 522
    maximum = 0.8
    image, draw = canvas(
        "2025 radiation error · lower is better",
        "MAE · kWh per square metre per day · common targets, zero-based scale",
    )
    grid(
        draw,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        x_ticks=[],
        y_ticks=[(v / maximum, f"{v:.1f}") for v in (0, 0.2, 0.4, 0.6, 0.8)],
    )
    for i, name in enumerate(names):
        value = maes[name]
        x = 250 + i * 276
        bar_top = bottom - (bottom - top) * value / maximum
        draw.rounded_rectangle((x, bar_top, x + 132, bottom), radius=13, fill=color(SERIES[name]))
        draw.text(
            (x + 66, bar_top - 26), f"{value:.3f}", fill=color(NAVY), font=font(21), anchor="ms"
        )
        draw.text((x + 66, bottom + 32), name, fill=color(NAVY), font=font(19), anchor="mt")
    return image


def reserve_chart(rows: list[dict[str, str]]) -> Image.Image:
    metrics = {row["approach"]: row for row in rows if row["scenario_id"] == "primary"}
    if set(metrics) != {"ridge", "persistence", "seasonal_mean"}:
        raise ValueError("expected Ridge, persistence, and seasonal mean primary rows")
    points = {
        "Persistence": (
            float(metrics["persistence"]["mean_simulated_shortfall_kwh_day"]),
            float(metrics["persistence"]["mean_excess_prepared_reserve_kwh_day"]),
        ),
        "Seasonal mean": (
            float(metrics["seasonal_mean"]["mean_simulated_shortfall_kwh_day"]),
            float(metrics["seasonal_mean"]["mean_excess_prepared_reserve_kwh_day"]),
        ),
        "Ridge": (
            float(metrics["ridge"]["mean_simulated_shortfall_kwh_day"]),
            float(metrics["ridge"]["mean_excess_prepared_reserve_kwh_day"]),
        ),
    }
    left, top, right, bottom = 192, 178, 1094, 516
    x_max, y_max = 1.0, 0.5
    image, draw = canvas(
        "Reserve trade-off · lower-left is preferred",
        "Mean simulated shortfall vs mean excess prepared reserve · kWh/day",
    )
    draw.text(
        (left, 143), "Mean excess reserve · down is preferred", fill=color(MUTED), font=font(17)
    )
    grid(
        draw,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        x_ticks=[(v / x_max, f"{v:.1f}") for v in (0, 0.2, 0.4, 0.6, 0.8, 1)],
        y_ticks=[(v / y_max, f"{v:.1f}") for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)],
    )
    for name, (shortfall, excess) in points.items():
        x = left + (right - left) * shortfall / x_max
        y = bottom - (bottom - top) * excess / y_max
        draw.ellipse(
            (x - 12, y - 12, x + 12, y + 12),
            fill=color(SERIES[name]),
            outline=color(OFF_WHITE),
            width=3,
        )
        draw.text((x + 19, y - 16), name, fill=color(NAVY), font=font(19))
        draw.text(
            (x + 19, y + 9), f"{shortfall:.3f}, {excess:.3f}", fill=color(MUTED), font=font(16)
        )
    draw.text(
        ((left + right) // 2, 595),
        "Mean simulated shortfall · toward left is preferred",
        fill=color(NAVY),
        font=font(19),
        anchor="mm",
    )
    return image


def main() -> None:
    predictions = read_rows(RUN / "predictions.csv")
    metrics = read_rows(RUN / "decision_metrics.csv")
    validate_target_dates(predictions)
    ASSETS.mkdir(parents=True, exist_ok=True)
    charts = {
        "forecast-comparison.png": january_chart(predictions),
        "forecast-error.png": mae_chart(predictions),
        "reserve-tradeoff.png": reserve_chart(metrics),
    }
    for filename, image in charts.items():
        image.save(ASSETS / filename, format="PNG", optimize=True)
    print(
        f"Generated {len(charts)} PNG charts from {len(predictions)} prediction rows "
        f"and {len(metrics)} decision metric rows."
    )


if __name__ == "__main__":
    main()
