from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ai_models import embed_texts, generate_text

MIN_RELEVANCE_SCORE = 0.08
RELATED_SOURCE_SCORE = 0.18
DEFAULT_TOP_K = 3


def knowledge_files(path: Path) -> list[Path]:
    if path.is_dir():
        root = path
        files = sorted([*root.glob("*.txt"), *root.glob("*.md")])
        return files

    files = sorted([*path.parent.glob("*.txt"), *path.parent.glob("*.md")])
    return [file for file in files if file.exists()]


def parse_qa_blocks(path: Path) -> list[dict[str, str]]:
    blocks = path.read_text(encoding="utf-8").split("\n\n")
    records = []
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 2 and lines[0].lower().startswith("q:"):
            question = lines[0].replace("Q:", "", 1).replace("q:", "", 1).strip()
            answer_lines = [
                line.replace("A:", "", 1).replace("a:", "", 1).strip()
                for line in lines[1:]
                if not line.startswith("#")
            ]
            answer = " ".join(answer_lines)
            records.append(
                {
                    "id": f"{path.stem}-{index}",
                    "question": question,
                    "answer": answer,
                    "source_file": path.name,
                }
            )
    return records


def load_faq(path: Path) -> list[dict[str, str]]:
    faqs = []
    for file in knowledge_files(path):
        faqs.extend(parse_qa_blocks(file))
    return faqs


def retrieve_with_embeddings(faq_df: pd.DataFrame, question: str, top_k: int) -> tuple[list[dict[str, str | float]], str]:
    text = (faq_df["question"] + " " + faq_df["answer"]).tolist()
    embedding_result = embed_texts([question] + text)
    vectors = embedding_result["vectors"]
    if not vectors:
        return [], embedding_result["mode"]

    query_vector = np.array(vectors[0], dtype=float).reshape(1, -1)
    matrix = np.array(vectors[1:], dtype=float)
    scores = cosine_similarity(query_vector, matrix).ravel()
    scored = faq_df.copy()
    scored["similarity"] = scores
    retrieved = scored.sort_values("similarity", ascending=False).head(top_k)
    return retrieved[["id", "question", "answer", "source_file", "similarity"]].to_dict("records"), "embedding"


def retrieve_with_tfidf(faq_df: pd.DataFrame, question: str, top_k: int) -> list[dict[str, str | float]]:
    text = faq_df["question"] + " " + faq_df["answer"]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text)
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).ravel()
    scored = faq_df.copy()
    scored["similarity"] = scores
    retrieved = scored.sort_values("similarity", ascending=False).head(top_k)
    return retrieved[["id", "question", "answer", "source_file", "similarity"]].to_dict("records")


def retrieve_context(
    faq_path: Path,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    retrieval_mode: str = "auto",
) -> tuple[list[dict[str, str | float]], str]:
    faqs = load_faq(faq_path)
    if not faqs or not question.strip():
        return [], "empty"

    faq_df = pd.DataFrame(faqs)
    if retrieval_mode in {"auto", "embedding"}:
        retrieved, mode = retrieve_with_embeddings(faq_df, question, top_k)
        if retrieved or retrieval_mode == "embedding":
            return retrieved, mode

    return retrieve_with_tfidf(faq_df, question, top_k), "tfidf"


def build_rag_prompt(question: str, contexts: list[dict[str, str | float]]) -> str:
    context_text = "\n\n".join(
        (
            f"Source {index} ({context['id']}, score={context['similarity']:.3f})\n"
            f"Question: {context['question']}\n"
            f"Answer: {context['answer']}"
        )
        for index, context in enumerate(contexts, start=1)
    )
    return (
        "You are a careful e-commerce support assistant for a capstone demo.\n"
        "Answer the customer using only the retrieved FAQ context below.\n"
        "If the context is not enough, say what is missing and suggest contacting official support.\n"
        "Keep the response concise, practical, and transparent that this is a demo system.\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Customer question: {question}\n\n"
        "Grounded answer:"
    )


def generate_grounded_response(
    question: str,
    contexts: list[dict[str, str | float]],
    model: str | None = None,
) -> tuple[str, str, str, str]:
    if not contexts:
        return "Please enter a support question so I can search the FAQ knowledge base.", "fallback", model or "", ""

    best = contexts[0]
    best_score = float(best["similarity"])
    if best_score < MIN_RELEVANCE_SCORE:
        return (
            "I could not find enough relevant FAQ context to answer that confidently. "
            "For a real store, please contact the official support channel with your order details.",
            "fallback",
            model or "",
            "",
        )

    prompt = build_rag_prompt(question, contexts)
    live_answer = generate_text(prompt, model=model, temperature=0.15)
    if live_answer["text"]:
        return live_answer["text"], live_answer["mode"], live_answer["model"], ""

    supporting = "; ".join(
        str(context["question"])
        for context in contexts[1:]
        if float(context["similarity"]) >= RELATED_SOURCE_SCORE
        and float(context["similarity"]) >= best_score * 0.65
    )
    best_question = str(best["question"]).rstrip(".?!")
    response = (
        f"Based on the retrieved FAQ: {best['answer']} "
        f"This answer is grounded in the closest support topic: {best_question}."
    )
    if supporting:
        response += f" Related retrieved topics: {supporting}."
    return response, "local_grounded", live_answer["model"], live_answer["error"]


def answer_question(
    faq_path: Path,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    model: str | None = None,
    retrieval_mode: str = "auto",
) -> dict[str, Any]:
    contexts, retrieval_used = retrieve_context(faq_path, question, top_k=top_k, retrieval_mode=retrieval_mode)
    answer, mode, model_used, generation_error = generate_grounded_response(question, contexts, model=model)
    best_context = contexts[0] if contexts else {"question": "", "similarity": 0.0}
    return {
        "question": best_context["question"],
        "answer": answer,
        "similarity": float(best_context["similarity"]),
        "sources": contexts,
        "prompt": build_rag_prompt(question, contexts) if contexts else "",
        "generation_mode": mode,
        "generation_error": generation_error,
        "retrieval_mode": retrieval_used,
        "model": model_used,
    }
