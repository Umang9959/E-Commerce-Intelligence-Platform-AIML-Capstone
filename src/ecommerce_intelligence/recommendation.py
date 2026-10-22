from __future__ import annotations

import pandas as pd

from .data_processing import positive_sales


def recommend_products(transactions: pd.DataFrame, selected_stock_code: str, top_n: int = 5) -> pd.DataFrame:
    sales_df = positive_sales(transactions)
    selected_stock_code = str(selected_stock_code)
    invoices = sales_df.loc[sales_df["stock_code"] == selected_stock_code, "invoice_no"].unique()

    popularity = sales_df.groupby(["stock_code", "description"]).agg(
        popularity=("quantity", "sum"),
        total_sales=("sales", "sum"),
    ).reset_index()

    if len(invoices) == 0:
        return popularity.sort_values(["popularity", "total_sales"], ascending=False).head(top_n)

    related = sales_df[sales_df["invoice_no"].isin(invoices)]
    related = related[related["stock_code"] != selected_stock_code]
    scores = related.groupby(["stock_code", "description"]).agg(
        co_purchase_count=("invoice_no", "nunique"),
        popularity=("quantity", "sum"),
    ).reset_index()

    if scores.empty:
        return popularity[popularity["stock_code"] != selected_stock_code].head(top_n)

    return scores.sort_values(["co_purchase_count", "popularity"], ascending=False).head(top_n)

