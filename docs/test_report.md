# Test Report

Use these checks before the final demo:

| Test | Input | Expected Result |
| --- | --- | --- |
| Active customer | A customer with recent purchases | Lower churn risk |
| Inactive customer | A customer with high recency | Higher churn risk |
| Recommendation | Product with historical co-purchases | Top related products shown |
| Search | A phrase like christmas or heart | Relevant products ranked first |
| Forecast | Forecast page opened | Historical and future values shown |
| Support RAG | How do I return a product? | Grounded answer generated with retrieved FAQ sources |
| Support fallback | Question unrelated to the FAQ | Low-confidence fallback shown instead of an invented answer |
| AI Studio | Generate executive brief | Business summary generated from dashboard metrics |
| Model comparison | Select multiple MaaS models | Each selected model produces a comparable strategy answer or offline fallback |
| Edge case | Empty query or unknown product | Friendly result without crashing |
