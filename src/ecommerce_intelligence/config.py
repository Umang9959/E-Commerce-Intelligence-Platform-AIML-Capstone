from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
RESULTS_DIR = OUTPUTS_DIR / "results"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

RAW_WORKBOOK = RAW_DIR / "online_retail_II.xlsx"
SAMPLE_RAW = RAW_DIR / "sample_online_retail.csv"

CLEANED_TRANSACTIONS = PROCESSED_DIR / "cleaned_transactions.csv"
CUSTOMER_FEATURES = PROCESSED_DIR / "customer_features.csv"
PRODUCT_FEATURES = PROCESSED_DIR / "product_features.csv"
SALES_TIME_SERIES = PROCESSED_DIR / "sales_time_series.csv"

CHURN_MODEL = MODELS_DIR / "churn_model.pkl"
CHURN_SCALER = MODELS_DIR / "churn_scaler.pkl"
CHURN_FEATURES = MODELS_DIR / "churn_features.pkl"
MODEL_RESULTS = RESULTS_DIR / "model_comparison.csv"
FORECAST_RESULTS = RESULTS_DIR / "forecast_metrics.json"

