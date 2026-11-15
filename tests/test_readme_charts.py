from datetime import date

import pytest
from scripts.generate_readme_charts import (
    RUN,
    radiation_mae_values,
    read_rows,
    validate_target_dates,
)


def test_frozen_2025_target_dates_are_complete_and_consecutive() -> None:
    rows = read_rows(RUN / "predictions.csv")

    dates = validate_target_dates(rows)

    assert len(dates) == 365
    assert dates[0] == date(2025, 1, 1)
    assert dates[-1] == date(2025, 12, 31)


def test_frozen_2025_mae_values_match_the_holdout_report() -> None:
    metrics = radiation_mae_values(read_rows(RUN / "predictions.csv"))

    assert metrics["Ridge"] == pytest.approx(0.5387915262)
    assert metrics["Persistence"] == pytest.approx(0.5885583562)
    assert metrics["Seasonal mean"] == pytest.approx(0.7119432544)


def test_chart_dates_reject_gaps() -> None:
    with pytest.raises(ValueError, match="consecutive"):
        validate_target_dates(
            [
                {"target_date": "2025-01-01"},
                {"target_date": "2025-01-03"},
            ]
        )
