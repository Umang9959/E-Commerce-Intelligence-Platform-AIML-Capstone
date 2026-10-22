from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config


def forecast_sales(sales_series: pd.DataFrame, periods: int = 8) -> tuple[pd.DataFrame, dict]:
    data = sales_series.copy()
    data["period"] = pd.to_datetime(data["period"])
    data = data.sort_values("period")

    if len(data) < 8:
        raise ValueError("Forecasting needs at least 8 time periods.")

    validation_size = min(8, max(2, len(data) // 5))
    train = data.iloc[:-validation_size]
    valid = data.iloc[-validation_size:]

    model = ExponentialSmoothing(train["sales"], trend="add", seasonal=None, initialization_method="estimated")
    fitted = model.fit(optimized=True)
    valid_pred = fitted.forecast(validation_size)

    metrics = {
        "mae": float(mean_absolute_error(valid["sales"], valid_pred)),
        "rmse": float(np.sqrt(mean_squared_error(valid["sales"], valid_pred))),
        "validation_periods": int(validation_size),
    }

    final_model = ExponentialSmoothing(data["sales"], trend="add", seasonal=None, initialization_method="estimated")
    final_fit = final_model.fit(optimized=True)
    future_values = final_fit.forecast(periods)

    last_period = data["period"].max()
    future_periods = pd.date_range(last_period + pd.Timedelta(days=7), periods=periods, freq="W")
    forecast = pd.DataFrame({"period": future_periods, "forecast_sales": future_values.values})

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FORECAST_RESULTS.write_text(json.dumps(metrics, indent=2))
    return forecast, metrics

