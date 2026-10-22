from __future__ import annotations

import json

import pandas as pd

from . import config


def data_quality_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"check": "Total rows", "value": len(transactions)},
        {"check": "Known customers", "value": int(transactions["customer_id"].notna().sum())},
        {"check": "Missing customers", "value": int(transactions["customer_id"].isna().sum())},
        {"check": "Cancellation or return rows", "value": int(transactions["is_cancellation"].sum())},
        {"check": "Unique products", "value": int(transactions["stock_code"].nunique())},
        {"check": "Unique invoices", "value": int(transactions["invoice_no"].nunique())},
    ]
    return pd.DataFrame(rows)


def load_model_results() -> pd.DataFrame:
    if config.MODEL_RESULTS.exists():
        return pd.read_csv(config.MODEL_RESULTS)
    return pd.DataFrame()


def load_forecast_metrics() -> dict:
    if config.FORECAST_RESULTS.exists():
        return json.loads(config.FORECAST_RESULTS.read_text())
    return {}

