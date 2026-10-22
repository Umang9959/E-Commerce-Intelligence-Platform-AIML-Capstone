from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config


FEATURE_COLUMNS = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "total_quantity",
    "product_diversity",
]


def train_churn_models(customer_features: pd.DataFrame) -> pd.DataFrame:
    X = customer_features[FEATURE_COLUMNS].fillna(0)
    y = customer_features["churned"].astype(int)

    if y.nunique() < 2:
        raise ValueError("Churn training needs both churned and active customers.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    rows = []
    best_name = None
    best_f1 = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        rows.append(row)
        if row["f1_score"] > best_f1:
            best_name = name
            best_f1 = row["f1_score"]
            best_model = model

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    results["final_role"] = results["model"].apply(lambda value: "Selected" if value == best_name else "Comparison")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.CHURN_MODEL)
    joblib.dump(FEATURE_COLUMNS, config.CHURN_FEATURES)
    results.to_csv(config.MODEL_RESULTS, index=False)
    return results


def load_churn_model():
    model = joblib.load(config.CHURN_MODEL)
    features = joblib.load(config.CHURN_FEATURES)
    return model, features


def risk_label(probability: float) -> str:
    if probability >= 0.65:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"


def explain_customer(row: pd.Series, probability: float) -> str:
    reasons = []
    if row["recency"] > 90:
        reasons.append("last purchase was a long time ago")
    else:
        reasons.append("customer purchased recently")
    if row["frequency"] <= 1:
        reasons.append("low repeat purchase activity")
    else:
        reasons.append("repeat purchase activity is visible")
    if row["monetary"] > 0:
        reasons.append("customer has positive purchase value")
    return f"Risk is {risk_label(probability)} because " + ", ".join(reasons) + "."


def predict_customer(customer_row: pd.Series) -> dict:
    model, features = load_churn_model()
    values = pd.to_numeric(customer_row[features], errors="coerce").fillna(0)
    X = pd.DataFrame([values.to_dict()])
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])
    else:
        probability = float(model.predict(X)[0])
    return {
        "probability": probability,
        "risk_level": risk_label(probability),
        "explanation": explain_customer(customer_row, probability),
    }
