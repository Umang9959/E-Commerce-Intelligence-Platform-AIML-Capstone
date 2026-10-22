# E-Commerce Intelligence Platform

Beginner-friendly AIML capstone project for the UCI Online Retail II dataset.

The project turns historical e-commerce transactions into:

- Customer churn prediction with three evaluated ML models
- Product recommendations using co-purchase history
- Sales forecasting from historical revenue
- Product catalog search using TF-IDF similarity
- FAQ-based customer support RAG with retrieved sources and grounded response generation
- AI Studio with MaaS/OpenAI-compatible model routing, executive brief generation, and multi-model comparison
- Monitoring outputs for model and feature quality
- A Streamlit dashboard for the final demo

## Dataset

Primary dataset: UCI Machine Learning Repository, Online Retail II.

- Page: https://archive.ics.uci.edu/dataset/502/online+retail
- Direct download used by the helper script: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- File expected after extraction: `online_retail_II.xlsx`
- License: Creative Commons Attribution 4.0 International
- Citation: Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

The dashboard uses the full `online_retail_II.xlsx` workbook as its dataset. Python reads both Excel sheets directly; no CSV conversion is needed. Put the workbook in `data/raw/online_retail_II.xlsx` before preparing the project data.

## Quick Start

1. Create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Make sure the full UCI workbook is at `data/raw/online_retail_II.xlsx`. If it is missing, download it with:

```bash
python scripts/download_dataset.py
```

4. Prepare the dashboard data from the workbook, then train the churn and forecast models:

```bash
python scripts/prepare_data.py
python scripts/train_models.py
```

5. (Optional) Configure MaaS as described in [MaaS / OpenAI-Compatible AI Setup](#maas--openai-compatible-ai-setup).

6. Start the dashboard:

```bash
streamlit run app.py
```

## MaaS / OpenAI-Compatible AI Setup

The project can run fully offline, but it becomes a live AI command center when you configure your model endpoint. Do not hardcode API keys in source files. Export them in your shell before starting Streamlit:

```bash
cd /home/labuser/Downloads/caps/caps
export MAAS_BASE_URL="https://genailab.tcs.in"
export MAAS_API_KEY="sk-evjWk7p1Ie-BvMhgaSxqXw"
export MAAS_CHAT_MODEL="genailab-maas-gpt-5.4"
export MAAS_EMBEDDING_MODEL="azure/genailab-maas-text-embedding-3-large"
export MAAS_VERIFY_SSL=false
streamlit run app.py
```

If your lab proxy requires disabled SSL verification, set `MAAS_VERIFY_SSL=false`. Use that only in the controlled lab environment.

## Customer Support RAG

The Support tab now runs a complete retrieval-augmented generation workflow:

1. Load support knowledge from every `.txt` and `.md` file in `knowledge/`.
2. Retrieve the most relevant FAQ chunks with semantic embeddings when the MaaS embedding model is configured.
3. Fall back to TF-IDF vectors and cosine similarity when offline.
3. Build a grounded prompt containing the user question and retrieved sources.
4. Generate a customer-facing answer from that retrieved context.
5. Show the answer, generation mode, similarity score, retrieved sources, and prompt evidence in the dashboard.

By default the project uses a deterministic local grounded generator so the capstone works offline. If the API key or model endpoint is unavailable, the app automatically falls back to local grounded generation. If retrieval confidence is too low, it gives a safe support fallback instead of pretending to know the answer.

The current knowledge corpus includes the original FAQ plus an expanded e-commerce support playbook with order, delivery, cancellation, return, refund, payment, product, account, privacy, analytics, and RAG workflow guidance.

## AI Studio

The AI Studio tab uses the dashboard's structured analytics as context and adds:

- A catalog of available chat and embedding models from the MaaS environment.
- Executive brief generation from customer, product, sales, churn, and forecast signals.
- Multi-model comparison so different models can answer the same business question.
- Offline fallback summaries when no API key is configured.

## Project Structure

```text
data/
  raw/
  processed/
models/
agents/
knowledge/
outputs/
docs/
scripts/
src/ecommerce_intelligence/
tests/
app.py
```

## Sprint Order

1. Prepare and clean the data.
2. Build customer features and train the three churn models.
3. Build product recommendations and catalog search.
4. Build sales forecasting and support RAG.
5. Integrate in Streamlit and capture evidence.

## Beginner Rule

Every feature in this project answers three questions:

- What is the input?
- What calculation or model happens?
- What output does the user see?






deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py