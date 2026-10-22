import pandas as pd

from src.ecommerce_intelligence.ai_insights import build_business_snapshot, compare_models, generate_executive_brief
from src.ecommerce_intelligence.ai_models import model_catalog
from src.ecommerce_intelligence.data_processing import (
    build_customer_features,
    build_product_features,
    build_sales_time_series,
    clean_transactions,
)
from src.ecommerce_intelligence.recommendation import recommend_products
from src.ecommerce_intelligence.search import search_products
from src.ecommerce_intelligence.support import answer_question, load_faq, retrieve_context


def tiny_raw_data():
    return pd.DataFrame(
        [
            [1, "A", "RED HEART MUG", 2, "2011-01-01", 3.0, 100, "United Kingdom"],
            [1, "B", "BLUE GIFT BAG", 1, "2011-01-01", 1.5, 100, "United Kingdom"],
            [2, "A", "RED HEART MUG", 1, "2011-02-01", 3.0, 101, "France"],
            [3, "C", "CHRISTMAS STAR", 4, "2011-05-01", 2.0, 102, "Germany"],
            ["C4", "A", "RED HEART MUG", -1, "2011-06-01", 3.0, 100, "United Kingdom"],
            [5, "M", "Manual", 1, "2011-06-02", 100.0, 103, "United Kingdom"],
        ],
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
    )


def test_cleaning_and_features():
    cleaned = clean_transactions(tiny_raw_data())
    customers = build_customer_features(cleaned, observation_days=30)
    products = build_product_features(cleaned)
    sales = build_sales_time_series(cleaned)

    assert cleaned["is_cancellation"].sum() == 1
    assert "Manual" not in set(cleaned["description"])
    assert len(customers) == 2
    assert set(customers["churned"]) == {1}
    assert len(products) == 3
    assert not sales.empty


def test_recommendation_and_search():
    cleaned = clean_transactions(tiny_raw_data())
    products = build_product_features(cleaned)

    recommendations = recommend_products(cleaned, "A")
    search_results = search_products(products, "heart")

    assert not recommendations.empty
    assert search_results.iloc[0]["description"] == "RED HEART MUG"


def test_support_rag_retrieves_sources_and_generates_grounded_answer(tmp_path, monkeypatch):
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    faq_path = tmp_path / "faq.txt"
    faq_path.write_text(
        "Q: How do I return a product?\n"
        "A: Customers can start a return from the order page and contact support with the order number.\n\n"
        "Q: Where is my order?\n"
        "A: Shipment tracking requires the official order-management system.\n",
        encoding="utf-8",
    )

    answer = answer_question(faq_path, "I need to return something")

    assert answer["generation_mode"] in {"local_grounded", "openai"}
    assert answer["sources"]
    assert answer["sources"][0]["question"] == "How do I return a product?"
    assert "return" in answer["answer"].lower()
    assert "Retrieved context" in answer["prompt"]


def test_support_rag_uses_fallback_for_unmatched_questions(tmp_path, monkeypatch):
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    faq_path = tmp_path / "faq.txt"
    faq_path.write_text(
        "Q: How do I return a product?\n"
        "A: Customers can start a return from the order page.\n",
        encoding="utf-8",
    )

    contexts, retrieval_mode = retrieve_context(faq_path, "What is your warehouse coffee machine brand?")
    answer = answer_question(faq_path, "What is your warehouse coffee machine brand?")

    assert contexts
    assert retrieval_mode in {"tfidf", "embedding", "offline", "error"}
    assert answer["generation_mode"] == "fallback"
    assert "could not find enough relevant FAQ context" in answer["answer"]


def test_support_loader_indexes_multiple_knowledge_files(tmp_path):
    faq_path = tmp_path / "faq.txt"
    extra_path = tmp_path / "returns.md"
    faq_path.write_text("Q: Where is my order?\nA: Check the official tracking page.\n", encoding="utf-8")
    extra_path.write_text("Q: How do exchanges work?\nA: Exchanges depend on stock and policy.\n", encoding="utf-8")

    records = load_faq(faq_path)
    questions = {record["question"] for record in records}

    assert "Where is my order?" in questions
    assert "How do exchanges work?" in questions
    assert all("source_file" in record for record in records)


def test_ai_model_catalog_and_offline_executive_brief(monkeypatch):
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cleaned = clean_transactions(tiny_raw_data())
    customers = build_customer_features(cleaned, observation_days=30)
    products = build_product_features(cleaned)
    sales = build_sales_time_series(cleaned)

    snapshot = build_business_snapshot(cleaned, customers, products, sales)
    result = generate_executive_brief("What should leadership do next?", snapshot, model="genailab-maas-gpt-5.4")
    catalog = model_catalog()

    assert snapshot["customers"] == 2
    assert catalog
    assert any(row["type"] == "embedding" for row in catalog)
    assert result["text"]
    assert result["mode"] in {"offline", "live", "error"}


def test_offline_model_comparison_uses_distinct_perspectives(monkeypatch):
    monkeypatch.delenv("MAAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cleaned = clean_transactions(tiny_raw_data())
    customers = build_customer_features(cleaned, observation_days=30)
    products = build_product_features(cleaned)
    sales = build_sales_time_series(cleaned)
    snapshot = build_business_snapshot(cleaned, customers, products, sales)

    rows = compare_models(
        "How should we reduce churn next week?",
        snapshot,
        ["genailab-maas-gpt-5.4", "azure/genailab-maas-gpt-4.1-mini", "gemini-2.5-pro"],
    )
    answers = {row["answer"] for row in rows}

    assert len(rows) == 3
    assert len(answers) == 3
    assert all("Offline answer for your question" in row["answer"] for row in rows)
