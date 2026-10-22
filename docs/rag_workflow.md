# Customer Support RAG Workflow

The support assistant uses retrieval-augmented generation instead of returning a static FAQ answer directly.

## Flow

1. The user asks a support question in the Streamlit Support tab.
2. `load_faq` reads Q/A records from every `.txt` and `.md` file in the `knowledge/` folder.
3. `retrieve_context` uses the MaaS embedding model for semantic vector retrieval when credentials are configured.
4. If live embeddings are unavailable, `retrieve_context` creates TF-IDF vectors for the FAQ records and ranks them with cosine similarity.
5. `build_rag_prompt` creates a grounded prompt containing the top retrieved FAQ sources.
6. `generate_grounded_response` generates the final response:
   - Uses MaaS/OpenAI-compatible chat generation when `MAAS_API_KEY` or `OPENAI_API_KEY` is available.
   - Uses a deterministic local grounded generator when the project is offline.
   - Uses a safe fallback when the top retrieval score is too low.
7. The dashboard displays the generated answer, selected model, retrieval mode, top similarity score, retrieved sources, and prompt evidence.

## Files

- `src/ecommerce_intelligence/support.py`: RAG retrieval, prompt building, generation, and fallback logic.
- `src/ecommerce_intelligence/ai_models.py`: MaaS/OpenAI-compatible model catalog, client routing, chat generation, and embeddings.
- `src/ecommerce_intelligence/ai_insights.py`: Executive AI Studio prompts and offline business summaries.
- `knowledge/faq.txt`: Original support FAQ.
- `knowledge/ecommerce_support_playbook.md`: Expanded e-commerce support, policy, AI, and dashboard knowledge corpus.
- `app.py`: Streamlit Support RAG and AI Studio tabs with model controls and evidence.
- `tests/test_pipeline.py`: Tests for retrieval, grounded generation, and low-confidence fallback.

## Demo Notes

The offline generator is included so the capstone can run in classrooms without external API access. It still follows the RAG workflow because it retrieves context, builds a prompt, and generates a grounded customer-facing response from that context. When an API key is configured, the same prompt is sent to the selected model for a more natural generated response.
