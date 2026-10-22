from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.ecommerce_intelligence import config
from src.ecommerce_intelligence.ai_insights import (
    build_business_snapshot,
    compare_models,
    generate_executive_brief,
)
from src.ecommerce_intelligence.ai_models import KNOWN_CHAT_MODELS, generate_text, get_ai_config, model_catalog
from src.ecommerce_intelligence.churn import predict_customer
from src.ecommerce_intelligence.forecasting import forecast_sales
from src.ecommerce_intelligence.monitoring import (
    data_quality_summary,
    load_forecast_metrics,
    load_model_results,
)
from src.ecommerce_intelligence.recommendation import recommend_products
from src.ecommerce_intelligence.search import search_products
from src.ecommerce_intelligence.support import answer_question


st.set_page_config(page_title="E-Commerce Intelligence Platform", layout="wide")

PALETTE = ["#4FD1C5", "#F08A5D", "#F6C85F", "#9B8CFF", "#68D391", "#F687B3", "#94A3B8"]
BACKGROUND = "#0E151B"
PANEL = "#151F29"
TEXT = "#E6EDF3"
MUTED_TEXT = "#AAB8C5"
GRID = "#2A3744"
ACCENT = "#4FD1C5"
ACCENT_WARM = "#F08A5D"


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79, 209, 197, 0.14), transparent 30rem),
            radial-gradient(circle at top right, rgba(240, 138, 93, 0.12), transparent 28rem),
            linear-gradient(180deg, #0b1117 0%, #111922 52%, #0e151b 100%);
        color: #e6edf3;
    }

    header[data-testid="stHeader"] {
        position: relative;
        background: linear-gradient(90deg, #0b1117 0%, #111922 58%, #0e151b 100%);
        border-bottom: 1px solid rgba(79, 209, 197, 0.22);
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.18);
    }

    header[data-testid="stHeader"]::after {
        content: "E-Commerce Intelligence Platform";
        position: absolute;
        top: 50%;
        left: 4.25rem;
        transform: translateY(-50%);
        color: #f3f7fb;
        font-size: 1.05rem;
        font-weight: 760;
        letter-spacing: -0.02em;
        white-space: nowrap;
        pointer-events: none;
    }

    header[data-testid="stHeader"] button {
        color: #d7e0ea !important;
    }

    header[data-testid="stHeader"] button:hover {
        background: rgba(79, 209, 197, 0.12) !important;
        color: #f7fbff !important;
    }

    @media (max-width: 560px) {
        header[data-testid="stHeader"]::after {
            left: 3.25rem;
            font-size: 0.82rem;
        }
    }

    .block-container {
        max-width: 1440px;
        padding: 2.1rem clamp(1.25rem, 3vw, 3rem) 3.5rem;
    }

    h1 {
        color: #f7fbff;
        font-size: clamp(2rem, 3vw, 2.8rem);
        font-weight: 760;
        letter-spacing: -0.035em;
    }

    h2, h3 {
        color: #e6edf3;
        letter-spacing: 0;
    }

    p, li, label, .stMarkdown, [data-testid="stMarkdownContainer"] {
        color: #d7e0ea;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24, 37, 48, 0.96), rgba(16, 26, 35, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-top: 3px solid #4fd1c5;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
        min-height: 112px;
    }

    [data-testid="stMetricLabel"] {
        color: #aab8c5;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #f7fbff;
        font-weight: 800;
    }

    [data-testid="stMetricValue"] > div {
        font-size: clamp(1.2rem, 1.7vw, 1.65rem);
        line-height: 1.1;
        white-space: nowrap;
    }

    div[data-testid="stDataFrame"] {
        max-width: min(100%, 1180px);
        margin: 0 auto;
        background: #111a22 !important;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.14);
    }

    [data-testid="stDataFrame"] canvas {
        background: #111a22 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        align-items: stretch;
        background: rgba(8, 13, 19, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        gap: 0.35rem;
        padding: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        align-items: center;
        background: transparent;
        border: 1px solid transparent;
        border-radius: 7px;
        color: #aab8c5;
        display: flex;
        font-weight: 700;
        justify-content: center;
        min-height: 2.65rem;
        padding: 0.6rem 1rem;
        transition:
            background-color 160ms ease,
            border-color 160ms ease,
            color 160ms ease,
            transform 160ms ease;
        white-space: nowrap;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 209, 197, 0.10);
        border-color: rgba(79, 209, 197, 0.20);
        color: #f7fbff;
        transform: translateY(-1px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(79, 209, 197, 0.24), rgba(79, 209, 197, 0.12));
        border-color: rgba(79, 209, 197, 0.36);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
        color: #f7fbff;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    @keyframes tab-content-enter {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stTabs [role="tabpanel"]:not([hidden]) {
        animation: tab-content-enter 420ms cubic-bezier(0.2, 0.75, 0.25, 1) both;
    }

    @media (prefers-reduced-motion: reduce) {
        .stTabs [role="tabpanel"]:not([hidden]) {
            animation: none;
        }
    }

    /* Keep the complete AI Studio section in code, but remove its tab from navigation. */
    .stTabs [role="tab"]:nth-of-type(8) {
        display: none !important;
    }

    .stCaption, caption {
        color: #aab8c5;
    }

    /* Streamlit/BaseWeb paints the actual input element separately from its wrapper. */
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div,
    div[data-baseweb="select"] > div {
        background-color: #090e13 !important;
        border: 1px solid #354452 !important;
        border-radius: 8px !important;
        color: #f3f7fb !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    input,
    textarea {
        background-color: #090e13 !important;
        color: #f3f7fb !important;
        caret-color: #4fd1c5 !important;
        -webkit-text-fill-color: #f3f7fb !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #91a0ad !important;
        -webkit-text-fill-color: #91a0ad !important;
        opacity: 1 !important;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-baseweb="textarea"] * {
        color: #f3f7fb !important;
    }

    div[data-baseweb="select"] svg {
        fill: #aab8c5 !important;
    }

    [data-baseweb="popover"] > div,
    [role="listbox"],
    [role="option"] {
        background-color: #090e13 !important;
        color: #f3f7fb !important;
    }

    [role="option"]:hover,
    [aria-selected="true"][role="option"] {
        background-color: #17302f !important;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: #4fd1c5 !important;
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] {
        width: fit-content;
        max-width: 82%;
        background: #17232d;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px 18px 18px 5px;
        margin: 0.45rem auto 0.45rem 0;
        padding: 0.38rem 0.85rem;
        box-shadow: 0 5px 16px rgba(0, 0, 0, 0.13);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #075e54, #087f6e);
        border-color: rgba(79, 209, 197, 0.28);
        border-radius: 18px 18px 5px 18px;
        margin: 0.45rem 0 0.45rem auto;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        text-align: right;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageAvatar"] {
        display: none;
    }

    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.35rem;
        line-height: 1.55;
    }

    [data-testid="stChatMessageContent"] p:last-child {
        margin-bottom: 0;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 8px;
        font-weight: 700;
        min-height: 2.6rem;
    }

    .stButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: #123b3a;
        border: 1px solid #4fd1c5;
        color: #f3f7fb;
    }

    .stAlert {
        background-color: rgba(21, 31, 41, 0.92);
        border-radius: 8px;
    }

    .intro-splash {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: grid;
        width: 100vw;
        height: 100vh;
        place-items: center;
        overflow: hidden;
        background:
            radial-gradient(circle at 50% 44%, rgba(79, 209, 197, 0.18), transparent 34rem),
            radial-gradient(circle at 85% 18%, rgba(240, 138, 93, 0.11), transparent 26rem),
            linear-gradient(145deg, #080e13 0%, #101a23 55%, #0b1117 100%);
        animation: intro-splash-exit 5.6s ease-in forwards;
    }

    .intro-splash::before {
        content: "";
        position: absolute;
        width: min(70vw, 700px);
        aspect-ratio: 1;
        border: 1px solid rgba(79, 209, 197, 0.12);
        border-radius: 50%;
        box-shadow: 0 0 100px rgba(79, 209, 197, 0.08), inset 0 0 100px rgba(79, 209, 197, 0.04);
        animation: intro-orbit 5.6s ease-out both;
    }

    .intro-splash-content {
        position: relative;
        z-index: 1;
        max-width: 1100px;
        padding: 2rem;
        text-align: center;
        animation: intro-title-enter 900ms cubic-bezier(0.2, 0.75, 0.25, 1) both;
    }

    .intro-splash h1 {
        margin: 0;
        color: #f3f7fb;
        font-size: clamp(2.4rem, 7vw, 5.6rem);
        font-weight: 800;
        letter-spacing: -0.055em;
        line-height: 1.08;
        text-wrap: balance;
        text-shadow: 0 8px 45px rgba(0, 0, 0, 0.35);
    }

    .intro-title-rule {
        width: clamp(90px, 16vw, 180px);
        height: 3px;
        margin: 1.7rem auto 0;
        border-radius: 99px;
        background: linear-gradient(90deg, transparent, #4fd1c5, #f08a5d, transparent);
        box-shadow: 0 0 24px rgba(79, 209, 197, 0.45);
    }

    @keyframes intro-title-enter {
        from {
            opacity: 0;
            transform: translateY(22px) scale(0.98);
            filter: blur(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: blur(0);
        }
    }

    @keyframes intro-splash-exit {
        0%, 80% {
            opacity: 1;
            visibility: visible;
        }
        100% {
            opacity: 0;
            visibility: hidden;
        }
    }

    @keyframes intro-orbit {
        from {
            opacity: 0;
            transform: scale(0.82);
        }
        22% {
            opacity: 1;
        }
        to {
            opacity: 0.55;
            transform: scale(1.06);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .intro-splash,
        .intro-splash::before {
            animation-duration: 1.2s;
        }

        .intro-splash-content {
            animation: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("intro_splash_shown", False):
    st.session_state["intro_splash_shown"] = True
    st.markdown(
        """
        <div class="intro-splash" aria-hidden="true">
            <div class="intro-splash-content">
                <h1>E-Commerce Intelligence Platform</h1>
                <div class="intro-title-rule"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=14)
    ax.set_facecolor(PANEL)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_pie_chart(title: str, labels, values) -> None:
    chart_data = pd.DataFrame({"label": labels, "value": values})
    chart_data = chart_data[chart_data["value"] > 0]
    if chart_data.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    fig, ax = plt.subplots(figsize=(4.1, 3.55), facecolor=PANEL)
    fig.subplots_adjust(top=0.84, bottom=0.24, left=0.08, right=0.92)
    wedges, _texts, autotexts = ax.pie(
        chart_data["value"],
        labels=None,
        autopct=lambda percentage: f"{percentage:.1f}%" if percentage >= 7 else "",
        startangle=90,
        pctdistance=0.78,
        radius=0.78,
        colors=[PALETTE[i % len(PALETTE)] for i in range(len(chart_data))],
        wedgeprops={"width": 0.34, "edgecolor": PANEL, "linewidth": 2},
        textprops={"fontsize": 9, "color": TEXT, "fontweight": "bold"},
    )
    centre = plt.Circle((0, 0), 0.45, fc=PANEL)
    ax.add_artist(centre)
    ax.axis("equal")
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=12)
    fig.legend(
        wedges,
        chart_data["label"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        labelcolor=TEXT,
        fontsize=7.5,
        ncol=2,
    )
    for text in autotexts:
        text.set_color(TEXT)
    st.pyplot(fig, clear_figure=True, use_container_width=False)


def show_line_chart(title: str, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    plot_data = frame.copy()
    plot_data[x_col] = pd.to_datetime(plot_data[x_col])
    plot_data = plot_data.sort_values(x_col)

    fig, ax = plt.subplots(figsize=(9.4, 3.35), facecolor=PANEL)
    ax.plot(plot_data[x_col], plot_data[y_col], color=ACCENT, linewidth=2.2)
    ax.fill_between(plot_data[x_col], plot_data[y_col], color=ACCENT, alpha=0.14)
    style_axis(ax, title)
    ax.set_xlabel("")
    ax.set_ylabel("Sales", color=MUTED_TEXT)
    fig.autofmt_xdate(rotation=0)
    st.pyplot(fig, clear_figure=True, use_container_width=False)


def show_horizontal_bar_chart(title: str, labels, values, color: str = ACCENT) -> None:
    chart_data = pd.DataFrame({"label": labels, "value": values}).sort_values("value")
    chart_data["label"] = chart_data["label"].astype(str).str.slice(0, 42)

    fig_height = max(2.7, min(5.0, len(chart_data) * 0.34))
    fig, ax = plt.subplots(figsize=(9.4, fig_height), facecolor=PANEL)
    bars = ax.barh(chart_data["label"], chart_data["value"], color=color, alpha=0.92)
    style_axis(ax, title)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.grid(False, axis="y")
    ax.set_xlabel("Value", color=MUTED_TEXT)
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f" {width:,.0f}", va="center", fontsize=8, color=MUTED_TEXT)
    st.pyplot(fig, clear_figure=True, use_container_width=False)


def show_grouped_metric_chart(model_results: pd.DataFrame, metric_cols: list[str]) -> None:
    plot_data = model_results.set_index("model")[metric_cols]
    fig, ax = plt.subplots(figsize=(9.4, 3.65), facecolor=PANEL)
    plot_data.plot(kind="bar", ax=ax, color=PALETTE[: len(metric_cols)], width=0.78)
    style_axis(ax, "Churn Model Metric Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score", color=MUTED_TEXT)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=TEXT, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.xticks(rotation=0, ha="center")
    st.pyplot(fig, clear_figure=True, use_container_width=False)


def show_table(frame: pd.DataFrame) -> None:
    """Render compact tables with consistent dark colors and readable alignment."""
    display = frame.copy()
    display.columns = [str(column).replace("_", " ").strip().title() for column in display.columns]
    styled = display.style.set_properties(
        **{
            "background-color": "#111a22",
            "color": "#d7e0ea",
            "border-color": "#2a3744",
            "text-align": "left",
        }
    )
    numeric_columns = display.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        styled = styled.set_properties(subset=numeric_columns, **{"text-align": "right"})
    styled = styled.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#1b2a36"),
                    ("color", "#f3f7fb"),
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("border-color", "#354452"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("color", "#f3f7fb !important"),
                    ("-webkit-text-fill-color", "#f3f7fb !important"),
                    ("padding", "0.45rem 0.7rem"),
                    ("max-width", "260px"),
                ],
            },
        ]
    )
    # Use Streamlit's HTML table renderer so the Styler's cell text colors are
    # applied directly. The dataframe canvas can ignore CSS colors for cells.
    st.table(styled)


def top_n_with_other(series: pd.Series, n: int = 6) -> pd.Series:
    top = series.sort_values(ascending=False).head(n)
    other_value = series.sum() - top.sum()
    if other_value > 0:
        top.loc["Other"] = other_value
    return top


@st.cache_data
def load_data():
    required = [
        config.CLEANED_TRANSACTIONS,
        config.CUSTOMER_FEATURES,
        config.PRODUCT_FEATURES,
        config.SALES_TIME_SERIES,
    ]
    if not all(path.exists() for path in required):
        return None
    return {
        "transactions": pd.read_csv(config.CLEANED_TRANSACTIONS, parse_dates=["invoice_date"]),
        "customers": pd.read_csv(config.CUSTOMER_FEATURES),
        "products": pd.read_csv(config.PRODUCT_FEATURES),
        "sales": pd.read_csv(config.SALES_TIME_SERIES),
    }


st.caption("Beginner-friendly AIML dashboard using UCI Online Retail II transactions.")

data = load_data()
if data is None:
    st.warning("Processed data was not found. Run the setup commands in README.md first.")
    st.code(
        "python scripts/prepare_data.py\n"
        "python scripts/train_models.py\n"
        "streamlit run app.py"
    )
    st.stop()

transactions = data["transactions"]
customers = data["customers"]
products = data["products"]
sales = data["sales"]

tabs = st.tabs(["Dashboard", "Churn", "Recommend", "Search", "Forecast", "Support RAG", "Direct Chat", "AI Studio", "Monitoring"])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers", f"{customers['customer_id'].nunique():,}")
    col2.metric("Products", f"{products['stock_code'].nunique():,}")
    col3.metric("Invoices", f"{transactions['invoice_no'].nunique():,}")
    col4.metric("Total Sales", f"{transactions.loc[transactions['sales'] > 0, 'sales'].sum():,.0f}")

    st.subheader("Sales trend")
    sales_chart = sales.copy()
    show_line_chart("Weekly Sales Movement", sales_chart, "period", "sales")

    st.subheader("Business overview")
    positive_transactions = transactions[transactions["sales"] > 0].copy()
    left, middle, right = st.columns(3)
    with left:
        country_sales = top_n_with_other(positive_transactions.groupby("country")["sales"].sum(), n=5)
        show_pie_chart("Sales by Country", country_sales.index, country_sales.values)
    with middle:
        churn_counts = customers["churned"].map({0: "Returned", 1: "Churned"}).value_counts()
        show_pie_chart("Customer Churn Split", churn_counts.index, churn_counts.values)
    with right:
        return_counts = transactions["is_cancellation"].map({False: "Normal sales rows", True: "Returns / cancellations"}).value_counts()
        show_pie_chart("Transaction Type", return_counts.index, return_counts.values)

    st.subheader("Top products")
    top_products = products.sort_values("total_sales", ascending=False).head(10)
    show_horizontal_bar_chart("Top Products by Sales", top_products["description"], top_products["total_sales"], color=ACCENT_WARM)

with tabs[1]:
    st.subheader("Customer churn prediction")
    if not config.CHURN_MODEL.exists():
        st.warning("Train the churn model first: python scripts/train_models.py")
    else:
        customer_id = st.selectbox("Customer ID", customers["customer_id"].sort_values())
        row = customers[customers["customer_id"] == customer_id].iloc[0]
        prediction = predict_customer(row)
        col1, col2, col3 = st.columns(3)
        col1.metric("Churn probability", f"{prediction['probability']:.1%}")
        col2.metric("Risk level", prediction["risk_level"])
        col3.metric("Recency", f"{int(row['recency'])} days")
        st.write(prediction["explanation"])
        details = row.astype(str).to_frame("value")
        churn_field_labels = {
            "customer_id": "Customer ID",
            "recency": "Days Since Last Purchase",
            "frequency": "Number of Orders",
            "monetary": "Total Spend",
            "avg_order_value": "Average Order Value",
            "total_quantity": "Total Quantity Purchased",
            "product_diversity": "Unique Products Purchased",
            "country": "Country",
            "last_purchase": "Last Purchase Date",
            "churned": "Historical Churn Status",
        }
        details.index = [
            churn_field_labels.get(str(field), str(field).replace("_", " ").strip().title())
            for field in details.index
        ]
        details.index.name = "Customer Field"
        details.columns = ["Customer Value"]
        show_table(details)

with tabs[2]:
    st.subheader("Product recommendation")
    product_options = products[["stock_code", "description"]].drop_duplicates()
    labels = product_options.apply(lambda r: f"{r['stock_code']} - {r['description']}", axis=1).tolist()
    selected = st.selectbox("Select a product", labels)
    stock_code = selected.split(" - ", 1)[0]
    recs = recommend_products(transactions, stock_code)
    show_table(recs)
    st.caption("Recommendations are based on products that appeared in the same historical invoices.")

with tabs[3]:
    st.subheader("Product catalog search")
    query = st.text_input("Search phrase", value="christmas heart")
    results = search_products(products, query)
    show_table(results)

with tabs[4]:
    st.subheader("Sales forecast")
    periods = st.slider("Future weeks", min_value=4, max_value=16, value=8)
    try:
        forecast, metrics = forecast_sales(sales, periods=periods)
        combined = pd.concat(
            [
                sales.rename(columns={"sales": "value"}).assign(type="historical")[["period", "value", "type"]],
                forecast.rename(columns={"forecast_sales": "value"}).assign(type="forecast")[["period", "value", "type"]],
            ]
        )
        fig, ax = plt.subplots(figsize=(9.4, 3.45), facecolor=PANEL)
        for chart_type, chart_group in combined.groupby("type"):
            chart_group = chart_group.copy()
            chart_group["period"] = pd.to_datetime(chart_group["period"])
            color = ACCENT if chart_type == "historical" else ACCENT_WARM
            ax.plot(chart_group["period"], chart_group["value"], label=chart_type.title(), linewidth=2.2, color=color)
        style_axis(ax, "Historical Sales and Forecast")
        ax.set_xlabel("")
        ax.set_ylabel("Sales", color=MUTED_TEXT)
        ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
        fig.autofmt_xdate(rotation=0)
        st.pyplot(fig, clear_figure=True, use_container_width=False)
        metric_left, metric_middle, metric_right = st.columns(3)
        metric_left.metric("Forecast MAE", f"{metrics['mae']:,.0f}")
        metric_middle.metric("Forecast RMSE", f"{metrics['rmse']:,.0f}")
        metric_right.metric("Validation weeks", f"{metrics['validation_periods']}")
    except Exception as exc:
        st.warning(f"Forecast could not be created: {exc}")

with tabs[5]:
    st.subheader("Customer support RAG assistant")
    configured_support_model = get_ai_config().chat_model
    support_model_index = (
        KNOWN_CHAT_MODELS.index(configured_support_model)
        if configured_support_model in KNOWN_CHAT_MODELS
        else 0
    )
    with st.form("support_rag_form"):
        question = st.text_input("Ask a support question", value="How do I return a product?")
        rag_left, rag_middle, rag_right = st.columns([2, 1, 1])
        with rag_left:
            support_model = st.selectbox(
                "Generation model",
                KNOWN_CHAT_MODELS,
                index=support_model_index,
                key="support_rag_model_v2",
            )
        with rag_middle:
            retrieval_mode = st.selectbox("Retrieval", ["auto", "embedding", "tfidf"], index=0)
        with rag_right:
            top_k = st.slider("Retrieved sources", min_value=1, max_value=5, value=3)
        run_rag = st.form_submit_button("Search and answer", type="primary")

    if run_rag:
        with st.spinner(f"Retrieving support context and asking {support_model}…"):
            st.session_state["support_rag_result"] = answer_question(
                config.KNOWLEDGE_DIR / "faq.txt",
                question,
                top_k=top_k,
                model=support_model,
                retrieval_mode=retrieval_mode,
            )

    answer = st.session_state.get("support_rag_result")
    if answer:
        st.write(answer["answer"])
        st.caption(
            f"Generation mode: {answer['generation_mode']} | "
            f"Retrieval: {answer['retrieval_mode']} | "
            f"Model: {answer['model'] or support_model} | "
            f"Top source: {answer['question']} | similarity: {answer['similarity']:.3f}"
        )
        if answer.get("generation_error"):
            st.warning(
                "The selected MaaS model did not generate an answer, so this reply uses the local "
                f"grounded fallback. API detail: {answer['generation_error']}"
            )
        with st.expander("Retrieved sources and RAG prompt"):
            sources = pd.DataFrame(answer["sources"])
            if sources.empty:
                st.info("No sources were retrieved.")
            else:
                show_table(sources)
            st.code(answer["prompt"] or "No prompt was built because no usable question was provided.")

with tabs[6]:
    st.subheader("Direct model chat")
    st.caption("Send a prompt straight to your configured API. This chat does not retrieve or use knowledge-base content.")
    if not get_ai_config().is_configured:
        st.info("Configure MAAS_API_KEY or OPENAI_API_KEY to send messages to your API.")

    for turn in st.session_state.get("direct_chat_history", []):
        with st.chat_message("user"):
            st.markdown(turn["prompt"])
        with st.chat_message("assistant"):
            result = turn["response"]
            st.caption(f"{result['provider']} · {result['model']} · {result['mode']}")
            if result["error"]:
                st.error(result["error"])
                if "connection" in result["error"].lower() or "connect" in result["error"].lower():
                    st.info(
                        "The request did not reach the model API. Check that this computer can access "
                        "the MaaS host (including any required VPN or proxy) and that MAAS_BASE_URL "
                        "matches the endpoint provided by your lab. If the detailed error mentions "
                        "a certificate verification failure, follow your lab's TLS configuration instructions."
                    )
            else:
                st.markdown(result["text"])

    if st.session_state.get("direct_chat_history") and st.button("Clear chat", key="clear_direct_chat"):
        st.session_state.pop("direct_chat_history", None)
        st.rerun()

    configured_chat_model = get_ai_config().chat_model
    configured_model_index = (
        KNOWN_CHAT_MODELS.index(configured_chat_model)
        if configured_chat_model in KNOWN_CHAT_MODELS
        else 0
    )
    with st.form("direct_chat_form", clear_on_submit=True):
        prompt = st.text_area(
            "Message",
            placeholder="Type a message…",
            height=96,
            key="direct_chat_prompt_v3",
        )
        chat_model = st.selectbox(
            "Model",
            KNOWN_CHAT_MODELS,
            index=configured_model_index,
            key="direct_chat_model_v3",
        )
        send_message = st.form_submit_button(
            "Send",
            type="primary",
            disabled=not get_ai_config().is_configured,
        )

    if send_message:
        if not prompt.strip():
            st.warning("Enter a message before sending.")
        else:
            with st.spinner(f"Waiting for {chat_model}…"):
                response = generate_text(prompt.strip(), model=chat_model)
            st.session_state.setdefault("direct_chat_history", []).append(
                {"prompt": prompt.strip(), "response": response}
            )
            st.rerun()

with tabs[7]:
    st.subheader("AI commerce command center")
    forecast_metrics = load_forecast_metrics()
    snapshot = build_business_snapshot(transactions, customers, products, sales, forecast_metrics)
    ai_config = get_ai_config()

    status_left, status_middle, status_right = st.columns(3)
    status_left.metric("AI provider", ai_config.provider_label)
    status_middle.metric("Default chat model", ai_config.chat_model)
    status_right.metric("Embedding model", ai_config.embedding_model)
    if not ai_config.is_configured:
        st.warning(
            "AI Studio is in offline preview because no MAAS_API_KEY or OPENAI_API_KEY is configured. "
            "Answers are generated by the local dashboard analyst, not by the selected live model."
        )

    st.write("Available model catalog")
    show_table(pd.DataFrame(model_catalog()))

    st.write("Executive AI briefing")
    question = st.text_area(
        "Business question",
        value=(
            "What should the store leadership do next week to reduce churn, increase basket value, "
            "and improve customer support?"
        ),
        height=90,
    )
    primary_model = st.selectbox("Primary strategy model", KNOWN_CHAT_MODELS, index=0)
    if st.button("Generate executive brief", type="primary"):
        st.session_state["ai_studio_brief"] = generate_executive_brief(question, snapshot, model=primary_model)

    if "ai_studio_brief" in st.session_state:
        result = st.session_state["ai_studio_brief"]
        st.caption(f"Mode: {result['mode']} | Provider: {result['provider']} | Model: {result['model']}")
        if result["error"] and result["mode"] != "offline":
            st.warning(result["error"])
        st.write(result["text"])
        with st.expander("Prompt and structured business snapshot"):
            st.code(result["prompt"])

    st.write("Multi-model strategy comparison")
    selected_models = st.multiselect(
        "Models to compare",
        KNOWN_CHAT_MODELS,
        default=[KNOWN_CHAT_MODELS[0], KNOWN_CHAT_MODELS[5], KNOWN_CHAT_MODELS[10]],
    )
    if st.button("Compare selected models"):
        st.session_state["ai_studio_comparison"] = compare_models(question, snapshot, selected_models[:5])

    if "ai_studio_comparison" in st.session_state:
        rows = st.session_state["ai_studio_comparison"]
        for row in rows:
            with st.expander(f"{row['model']} | {row['mode']}", expanded=True):
                if row["error"] and row["mode"] != "offline":
                    st.warning(row["error"])
                st.write(row["answer"])

with tabs[8]:
    st.subheader("Monitoring and evidence")
    quality = data_quality_summary(transactions)
    model_results = load_model_results()
    forecast_metrics = load_forecast_metrics()

    st.write("Data quality summary")
    show_table(quality)
    show_horizontal_bar_chart("Data Quality Checks", quality["check"], quality["value"], color="#9B8CFF")

    st.write("Data health visuals")
    left, right = st.columns(2)
    with left:
        known_missing = pd.Series(
            {
                "Known Customer ID": int(transactions["customer_id"].notna().sum()),
                "Missing Customer ID": int(transactions["customer_id"].isna().sum()),
            }
        )
        show_pie_chart("Customer ID Completeness", known_missing.index, known_missing.values)
    with right:
        normal_return = transactions["is_cancellation"].map({False: "Normal sales rows", True: "Returns / cancellations"}).value_counts()
        show_pie_chart("Returns and Cancellations", normal_return.index, normal_return.values)

    st.write("Churn model comparison")
    if model_results.empty:
        st.info("Model results are not available yet. Run python scripts/train_models.py.")
    else:
        show_table(model_results)
        metric_cols = ["accuracy", "precision", "recall", "f1_score"]
        show_grouped_metric_chart(model_results, metric_cols)
        selected = model_results.loc[model_results["final_role"] == "Selected"].iloc[0]
        st.success(
            f"Selected churn model: {selected['model']} with F1-score {selected['f1_score']:.3f}."
        )

    st.write("Forecast monitoring")
    if forecast_metrics:
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecast MAE", f"{forecast_metrics['mae']:,.2f}")
        col2.metric("Forecast RMSE", f"{forecast_metrics['rmse']:,.2f}")
        col3.metric("Validation periods", f"{forecast_metrics['validation_periods']}")
    else:
        st.info("Forecast metrics are not available yet. Run python scripts/train_models.py.")
