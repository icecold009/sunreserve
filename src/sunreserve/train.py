"""Simple leakage-safe radiation baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sunreserve.config import ExperimentConfig
from sunreserve.features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class SeasonalMeanBaseline:
    """Target-month averages fitted only from training labels."""

    monthly_means: dict[int, float]
    training_global_mean: float
    training_rows: int

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if "target_date" not in rows:
            raise ValueError("prediction rows must contain target_date")
        months = pd.to_datetime(rows["target_date"]).dt.month
        predictions = months.map(self.monthly_means).fillna(self.training_global_mean)
        return pd.Series(
            np.maximum(predictions.to_numpy(dtype=float), 0.0),
            index=rows.index,
            name="seasonal_mean_prediction_kwh_m2_day",
        )


def _validated_training_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "target_date", TARGET_COLUMN}
    if not required <= set(rows.columns):
        raise ValueError(f"training rows must contain {', '.join(sorted(required))}")
    labels = set(rows["split"].dropna())
    if labels != {"train"}:
        raise ValueError("seasonal baseline must be fit from training rows only")
    if rows.empty:
        raise ValueError("cannot fit seasonal baseline on an empty training partition")
    if rows["target_date"].duplicated().any():
        raise ValueError("training rows contain duplicate target dates")
    target = pd.to_numeric(rows[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("training targets must be finite")
    if (target < 0).any():
        raise ValueError("training solar-radiation targets must be nonnegative")
    return rows.assign(
        target_date=pd.to_datetime(rows["target_date"]),
        _target_numeric=target,
    )


def fit_seasonal_mean(training_rows: pd.DataFrame) -> SeasonalMeanBaseline:
    """Fit target-calendar-month and global means from training labels only."""
    rows = _validated_training_rows(training_rows)
    months = rows["target_date"].dt.month
    target = rows["_target_numeric"]
    monthly = target.groupby(months).mean().to_dict()
    return SeasonalMeanBaseline(
        monthly_means={int(month): float(mean) for month, mean in monthly.items()},
        training_global_mean=float(target.mean()),
        training_rows=int(len(rows)),
    )


def predict_persistence(rows: pd.DataFrame) -> pd.Series:
    """Use radiation observed on the issue date as tomorrow's forecast."""
    column = "radiation_today_kwh_m2_day"
    if column not in rows:
        raise ValueError(f"prediction rows must contain {column}")
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("persistence inputs must be finite")
    return pd.Series(
        np.maximum(values, 0.0),
        index=rows.index,
        name="persistence_prediction_kwh_m2_day",
    )


def baseline_predictions(
    training_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    *,
    allow_final_test: bool = False,
) -> pd.DataFrame:
    """Predict both baselines on precisely the same eligible target-date rows."""
    if evaluation_rows.empty:
        raise ValueError("evaluation rows are empty")
    if "split" not in evaluation_rows:
        raise ValueError("evaluation rows must identify their split")
    evaluation_splits = set(evaluation_rows["split"].dropna())
    if len(evaluation_splits) != 1:
        raise ValueError("baseline evaluation must use exactly one named partition")
    if "test" in evaluation_splits and not allow_final_test:
        raise ValueError("final test predictions require explicit finalization")
    if evaluation_rows["target_date"].duplicated().any():
        raise ValueError("evaluation rows contain duplicate target dates")
    model = fit_seasonal_mean(training_rows)
    result = evaluation_rows.loc[:, ["issue_date", "target_date", TARGET_COLUMN]].copy()
    result["persistence_prediction_kwh_m2_day"] = predict_persistence(evaluation_rows)
    result["seasonal_mean_prediction_kwh_m2_day"] = model.predict(evaluation_rows)
    if not result["target_date"].equals(evaluation_rows["target_date"]):
        raise RuntimeError("baseline predictions do not align with evaluation dates")
    return result.reset_index(drop=True)


def fit_ridge(training_rows: pd.DataFrame, config: ExperimentConfig) -> Pipeline:
    """Fit imputation, scaling, and Ridge on training rows only."""
    _validated_training_rows(training_rows)
    missing_columns = set(FEATURE_COLUMNS) - set(training_rows.columns)
    if missing_columns:
        raise ValueError(
            f"Ridge training rows are missing features: {', '.join(sorted(missing_columns))}"
        )
    features = training_rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = features.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Ridge training features must not contain infinite values")
    if features.isna().all(axis=0).any():
        columns = features.columns[features.isna().all(axis=0)].tolist()
        raise ValueError(f"Ridge training features are entirely missing: {', '.join(columns)}")
    target = pd.to_numeric(training_rows[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=config.ridge_alpha,
                    solver="lsqr",
                    tol=1e-4,
                    max_iter=1000,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model


def predict_ridge(model: Pipeline, rows: pd.DataFrame) -> pd.Series:
    """Predict with a fitted training-only pipeline and clip below zero."""
    missing_columns = set(FEATURE_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise ValueError(
            f"Ridge prediction rows are missing features: {', '.join(sorted(missing_columns))}"
        )
    features = rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = features.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Ridge prediction features must not contain infinite values")
    predictions = model.predict(features)
    return pd.Series(
        np.maximum(predictions, 0.0),
        index=rows.index,
        name="ridge_prediction_kwh_m2_day",
    )
