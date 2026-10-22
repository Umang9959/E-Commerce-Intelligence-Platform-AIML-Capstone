from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ecommerce_intelligence import config
from src.ecommerce_intelligence.churn import train_churn_models
from src.ecommerce_intelligence.forecasting import forecast_sales


def main() -> None:
    customers = pd.read_csv(config.CUSTOMER_FEATURES)
    sales_series = pd.read_csv(config.SALES_TIME_SERIES)

    print("Training churn models")
    results = train_churn_models(customers)
    print(results.to_string(index=False))

    print("\nTraining sales forecast")
    forecast, metrics = forecast_sales(sales_series)
    forecast.to_csv(config.RESULTS_DIR / "sales_forecast.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
