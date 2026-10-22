from __future__ import annotations

import pandas as pd

from . import config


STANDARD_COLUMNS = {
    "Invoice": "invoice_no",
    "InvoiceNo": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "UnitPrice": "unit_price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}

NON_PRODUCT_STOCK_CODES = {
    "AMAZONFEE",
    "BANK CHARGES",
    "C2",
    "CRUK",
    "D",
    "DOT",
    "M",
    "POST",
    "S",
}
NON_PRODUCT_DESCRIPTIONS = {
    "AMAZON FEE",
    "BANK CHARGES",
    "CRUK COMMISSION",
    "DISCOUNT",
    "DOTCOM POSTAGE",
    "MANUAL",
    "POSTAGE",
    "SAMPLES",
}


def ensure_directories() -> None:
    for path in [
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.CHARTS_DIR,
        config.RESULTS_DIR,
        config.KNOWLEDGE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_transactions(use_sample: bool = False) -> pd.DataFrame:
    """Load either the full UCI workbook or the beginner sample CSV."""
    if use_sample:
        if not config.SAMPLE_RAW.exists():
            raise FileNotFoundError("Run python scripts/create_sample_data.py first.")
        return pd.read_csv(config.SAMPLE_RAW)

    if not config.RAW_WORKBOOK.exists():
        raise FileNotFoundError(
            "data/raw/online_retail_II.xlsx was not found. "
            "Run scripts/download_dataset.py or use --sample."
        )

    # Online Retail II has two sheets with the same structure.
    sheets = pd.read_excel(config.RAW_WORKBOOK, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        frame = frame.copy()
        frame["source_sheet"] = sheet_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={col: STANDARD_COLUMNS.get(col, col) for col in df.columns})
    expected = [
        "invoice_no",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
    ]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[expected].copy()


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(raw)
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["description"] = df["description"].fillna("Unknown Product").astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")

    df = df.dropna(subset=["invoice_no", "stock_code", "invoice_date", "quantity", "unit_price"])
    df = df.drop_duplicates()
    product_rows = ~(
        df["stock_code"].str.upper().isin(NON_PRODUCT_STOCK_CODES)
        | df["description"].str.upper().isin(NON_PRODUCT_DESCRIPTIONS)
    )
    df = df[product_rows].copy()

    df["is_cancellation"] = df["invoice_no"].str.upper().str.startswith("C") | (df["quantity"] <= 0)
    df["sales"] = df["quantity"] * df["unit_price"]
    df["invoice_date_only"] = df["invoice_date"].dt.date
    return df


def positive_sales(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["quantity"] > 0) & (df["unit_price"] > 0)].copy()


def build_customer_features(df: pd.DataFrame, observation_days: int = 90) -> pd.DataFrame:
    """Create customer features without leaking future behavior into training.

    Features are calculated only from purchases on or before the cutoff date.
    The churn target checks whether the customer returned during the future
    observation window after that cutoff date.
    """
    sales_df = positive_sales(df).dropna(subset=["customer_id"]).copy()
    if sales_df.empty:
        raise ValueError("No positive sales rows with customer_id are available.")

    max_date = sales_df["invoice_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=observation_days)
    historical = sales_df[sales_df["invoice_date"] <= cutoff_date].copy()
    future = sales_df[sales_df["invoice_date"] > cutoff_date].copy()

    if historical.empty or future.empty:
        raise ValueError("Not enough date range to create historical features and future churn target.")

    grouped = historical.groupby("customer_id")
    features = grouped.agg(
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("sales", "sum"),
        total_quantity=("quantity", "sum"),
        product_diversity=("stock_code", "nunique"),
        country=("country", lambda values: values.mode().iloc[0] if not values.mode().empty else "Unknown"),
    ).reset_index()
    features["recency"] = (cutoff_date - features["last_purchase"]).dt.days
    features["avg_order_value"] = features["monetary"] / features["frequency"].clip(lower=1)

    future_customers = set(future["customer_id"].unique())
    features["churned"] = (~features["customer_id"].isin(future_customers)).astype(int)
    return features[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "avg_order_value",
            "total_quantity",
            "product_diversity",
            "country",
            "churned",
        ]
    ]


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    sales_df = positive_sales(df).copy()
    products = sales_df.groupby(["stock_code", "description"]).agg(
        total_quantity=("quantity", "sum"),
        total_sales=("sales", "sum"),
        invoice_count=("invoice_no", "nunique"),
    ).reset_index()
    return products.sort_values("total_sales", ascending=False)


def build_sales_time_series(df: pd.DataFrame, frequency: str = "W") -> pd.DataFrame:
    sales_df = positive_sales(df).copy()
    sales_df = sales_df.set_index("invoice_date")
    series = sales_df["sales"].resample(frequency).sum().reset_index()
    series = series.rename(columns={"invoice_date": "period", "sales": "sales"})
    return series


def prepare_all(use_sample: bool = False) -> dict[str, pd.DataFrame]:
    ensure_directories()
    raw = load_raw_transactions(use_sample=use_sample)
    cleaned = clean_transactions(raw)
    customers = build_customer_features(cleaned)
    products = build_product_features(cleaned)
    sales_series = build_sales_time_series(cleaned)

    cleaned.to_csv(config.CLEANED_TRANSACTIONS, index=False)
    customers.to_csv(config.CUSTOMER_FEATURES, index=False)
    products.to_csv(config.PRODUCT_FEATURES, index=False)
    sales_series.to_csv(config.SALES_TIME_SERIES, index=False)

    return {
        "cleaned": cleaned,
        "customers": customers,
        "products": products,
        "sales_series": sales_series,
    }
