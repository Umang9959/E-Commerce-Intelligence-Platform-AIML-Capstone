from __future__ import annotations

from typing import Any

import pandas as pd

from .ai_models import describe_model, generate_text


def build_business_snapshot(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    sales: pd.DataFrame,
    forecast_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    positive = transactions[transactions["sales"] > 0].copy()
    top_products = products.sort_values("total_sales", ascending=False).head(5)
    top_countries = positive.groupby("country")["sales"].sum().sort_values(ascending=False).head(5)
    churn_rate = float(customers["churned"].mean()) if not customers.empty else 0.0

    return {
        "customers": int(customers["customer_id"].nunique()),
        "products": int(products["stock_code"].nunique()),
        "invoices": int(transactions["invoice_no"].nunique()),
        "total_sales": float(positive["sales"].sum()),
        "churn_rate": churn_rate,
        "return_rows": int(transactions["is_cancellation"].sum()),
        "top_products": top_products[["stock_code", "description", "total_sales"]].to_dict("records"),
        "top_countries": top_countries.to_dict(),
        "latest_sales_period": str(sales["period"].max()) if not sales.empty else "",
        "forecast_metrics": forecast_metrics or {},
    }


def build_insight_prompt(question: str, snapshot: dict[str, Any]) -> str:
    return (
        "You are the AI strategy layer for an e-commerce intelligence dashboard.\n"
        "Use only the structured business snapshot below. Give practical, decision-ready advice.\n"
        "Organize the answer into: Executive readout, Evidence, Recommended actions, Risks.\n\n"
        f"Business snapshot:\n{snapshot}\n\n"
        f"Business question: {question}"
    )


def question_focus(question: str) -> str:
    normalized = question.lower()
    if any(term in normalized for term in ["churn", "retention", "retain", "win back", "win-back"]):
        return "retention"
    if any(term in normalized for term in ["basket", "aov", "cross-sell", "upsell", "recommend"]):
        return "basket"
    if any(term in normalized for term in ["refund", "return", "cancel", "support", "complaint", "delivery", "order"]):
        return "support"
    if any(term in normalized for term in ["forecast", "sales", "revenue", "inventory", "stock"]):
        return "forecast"
    if any(term in normalized for term in ["risk", "monitor", "quality", "data"]):
        return "monitoring"
    return "general"


def model_perspective(model: str | None) -> str:
    if not model:
        return "Balanced offline analyst"

    description = describe_model(model)
    lowered = model.lower()
    if "gemini" in lowered:
        return "Long-context analyst: connects customer, product, support, and forecast signals"
    if "nano" in lowered or "mini" in lowered or "lite" in lowered:
        return "Fast operator: short, practical next steps with minimal detail"
    if "llama" in lowered or "maverick" in lowered:
        return "Open-model challenger: stresses assumptions, risks, and validation"
    if "reasoning" in lowered or "5.4" in lowered or "5.2" in lowered:
        return "Deep strategy analyst: prioritizes causes, tradeoffs, and action sequencing"
    return f"{description}: balanced business recommendations"


def local_executive_brief(snapshot: dict[str, Any], question: str, model: str | None = None) -> str:
    top_product = snapshot["top_products"][0] if snapshot["top_products"] else {}
    top_country = next(iter(snapshot["top_countries"]), "Unknown")
    churn_rate = snapshot["churn_rate"] * 100
    sales = snapshot["total_sales"]
    focus = question_focus(question)
    perspective = model_perspective(model)
    forecast_metrics = snapshot.get("forecast_metrics") or {}

    playbooks = {
        "retention": (
            "Recommended actions: Build a weekly high-risk customer list, target customers with high recency and low frequency, "
            "send win-back offers tied to top product categories, and monitor whether churn probability drops after campaigns."
        ),
        "basket": (
            "Recommended actions: Use co-purchase recommendations to create bundles, feature the strongest product as an anchor, "
            "test threshold discounts, and promote complementary items during checkout."
        ),
        "support": (
            "Recommended actions: Use Support RAG for order, refund, delivery, return, and cancellation questions, add missing policies "
            "to the knowledge base, and escalate live account-specific issues to official support."
        ),
        "forecast": (
            "Recommended actions: Use forecast error metrics to set confidence bands, plan inventory around leading countries and products, "
            "and review outliers before campaign or stock decisions."
        ),
        "monitoring": (
            "Recommended actions: Track missing customer IDs, return rows, model metrics, and forecast errors before trusting recommendations "
            "for production decisions."
        ),
        "general": (
            "Recommended actions: Combine churn targeting, co-purchase bundles, forecast-based planning, and Support RAG improvements into one weekly operating plan."
        ),
    }

    evidence_parts = [
        f"{snapshot['customers']:,} customers",
        f"{snapshot['products']:,} products",
        f"{snapshot['invoices']:,} invoices",
        f"{sales:,.0f} positive sales value",
        f"{churn_rate:.1f}% churn rate",
        f"{snapshot['return_rows']:,} return or cancellation rows",
    ]
    if forecast_metrics:
        evidence_parts.append(f"forecast MAE {forecast_metrics.get('mae', 'n/a')}")

    return (
        f"Offline answer for your question: {question}\n\n"
        f"Lens: {perspective}.\n\n"
        f"Executive readout: Focus on {focus}. The store has enough historical signal to act now: "
        f"{', '.join(evidence_parts)}.\n\n"
        f"Evidence: The leading product is {top_product.get('description', 'not available')} "
        f"and the leading market is {top_country}. These should anchor the first experiment unless your question is about a narrower segment.\n\n"
        f"{playbooks[focus]}\n\n"
        "Next 7 days: pick one measurable experiment, define the target customer/product group, launch the action, "
        "and compare churn, basket value, support resolution quality, and sales movement before and after.\n\n"
        "Limitation: This is the local offline analyst. Configure MAAS_API_KEY to get a true live answer from the selected model."
    )


def generate_executive_brief(
    question: str,
    snapshot: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    prompt = build_insight_prompt(question, snapshot)
    result = generate_text(prompt, model=model, temperature=0.25)
    if result["text"]:
        return {**result, "text": result["text"], "prompt": prompt}
    return {**result, "text": local_executive_brief(snapshot, question, model=model), "prompt": prompt}


def compare_models(question: str, snapshot: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        result = generate_executive_brief(question, snapshot, model=model)
        rows.append(
            {
                "model": result["model"],
                "mode": result["mode"],
                "provider": result["provider"],
                "error": result["error"],
                "answer": result["text"],
            }
        )
    return rows
