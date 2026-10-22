from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_products(products: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return pd.DataFrame(columns=["stock_code", "description", "similarity"])

    catalog = products.dropna(subset=["description"]).drop_duplicates("description").copy()
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(catalog["description"].astype(str))
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    result = catalog[["stock_code", "description"]].copy()
    result["similarity"] = scores
    return result.sort_values("similarity", ascending=False).head(top_n)

