from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


KNOWN_CHAT_MODELS = [
    "genailab-maas-gpt-5.4",
    "genailab-maas-gpt-5.2",
    "genailab-maas-gpt-5.1",
    "genailab-maas-gpt-5.0",
    "azure/genailab-maas-gpt-5-mini",
    "genailab-maas-gpt-4o",
    "azure/genailab-maas-gpt-4.1",
    "azure/genailab-maas-gpt-4.1-mini",
    "azure/genailab-maas-gpt-4.1-nano",
    "azure/genailab-maas-gpt-4o-mini",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "azure_ai/genailab-maas-Phi-4-reasoning",
    "genailab-maas-gpt-35-turbo",
]

KNOWN_EMBEDDING_MODELS = [
    "azure/genailab-maas-text-embedding-3-large",
]


@dataclass(frozen=True)
class AIModelConfig:
    base_url: str | None
    api_key: str | None
    chat_model: str
    embedding_model: str
    verify_ssl: bool

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_label(self) -> str:
        if self.base_url:
            return "MaaS / OpenAI-compatible"
        if self.api_key:
            return "OpenAI"
        return "Offline fallback"


def get_ai_config(chat_model: str | None = None, embedding_model: str | None = None) -> AIModelConfig:
    return AIModelConfig(
        base_url=os.getenv("MAAS_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("MAAS_API_KEY") or os.getenv("OPENAI_API_KEY"),
        chat_model=chat_model or os.getenv("MAAS_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "genailab-maas-gpt-4o",
        embedding_model=embedding_model
        or os.getenv("MAAS_EMBEDDING_MODEL")
        or "azure/genailab-maas-text-embedding-3-large",
        verify_ssl=(os.getenv("MAAS_VERIFY_SSL", "true").lower() != "false"),
    )


def model_catalog() -> list[dict[str, str]]:
    rows = []
    for model in KNOWN_CHAT_MODELS:
        rows.append({"model": model, "type": "chat", "best_for": describe_model(model)})
    for model in KNOWN_EMBEDDING_MODELS:
        rows.append({"model": model, "type": "embedding", "best_for": "Semantic retrieval and vector search"})
    return rows


def describe_model(model: str) -> str:
    name = model.lower()
    if "embedding" in name:
        return "Semantic retrieval"
    if "nano" in name or "mini" in name or "lite" in name:
        return "Fast routing and concise answers"
    if "reasoning" in name or "5.4" in name or "5.2" in name:
        return "Deep analysis and strategy"
    if "gemini" in name:
        return "Long-context synthesis"
    if "llama" in name or "maverick" in name:
        return "Open-model comparison"
    return "General generation"


def create_client(config: AIModelConfig):
    from openai import OpenAI

    if config.verify_ssl:
        http_client = None
    else:
        import httpx

        http_client = httpx.Client(verify=False)

    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if http_client:
        kwargs["http_client"] = http_client
    return OpenAI(**kwargs)


def generate_text(prompt: str, model: str | None = None, temperature: float = 0.2) -> dict[str, Any]:
    config = get_ai_config(chat_model=model)
    if not config.is_configured:
        return {
            "text": "",
            "mode": "offline",
            "model": config.chat_model,
            "provider": config.provider_label,
            "error": "Set MAAS_API_KEY or OPENAI_API_KEY to enable live generation.",
        }

    try:
        client = create_client(config)
        request: dict[str, Any] = {
            "model": config.chat_model,
            "messages": [
                {"role": "system", "content": "You are a precise, grounded AI analyst for an e-commerce capstone."},
                {"role": "user", "content": prompt},
            ],
        }
        # Some MaaS model groups reject temperature overrides. Leave the
        # provider's default untouched for GPT-5 and Gemini 2.5 models.
        model_name = config.chat_model.lower()
        if "gpt-5" not in model_name and not model_name.startswith("gemini-2.5"):
            request["temperature"] = temperature
        response = client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        return {
            "text": text.strip(),
            "mode": "live",
            "model": config.chat_model,
            "provider": config.provider_label,
            "error": "",
        }
    except Exception as exc:
        error_details = [str(exc).strip()]
        cause = exc.__cause__ or exc.__context__
        seen_causes = set()
        while cause is not None and id(cause) not in seen_causes and len(error_details) < 3:
            seen_causes.add(id(cause))
            cause_text = str(cause).strip()
            if cause_text:
                error_details.append(f"{type(cause).__name__}: {cause_text[:500]}")
            cause = cause.__cause__ or cause.__context__
        error_message = " | ".join(part for part in error_details if part) or type(exc).__name__
        return {
            "text": "",
            "mode": "error",
            "model": config.chat_model,
            "provider": config.provider_label,
            "error": error_message,
        }


def embed_texts(texts: list[str], model: str | None = None) -> dict[str, Any]:
    config = get_ai_config(embedding_model=model)
    if not config.is_configured:
        return {"vectors": [], "mode": "offline", "model": config.embedding_model, "error": "No API key configured."}

    try:
        client = create_client(config)
        response = client.embeddings.create(model=config.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return {"vectors": vectors, "mode": "live", "model": config.embedding_model, "error": ""}
    except Exception as exc:
        return {"vectors": [], "mode": "error", "model": config.embedding_model, "error": str(exc)}
