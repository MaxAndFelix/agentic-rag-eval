# P1 Build Plan — Agentic RAG + Eval Harness (2–3 days)

Corpus decision (Day 1, first thing): use the CSCI 544 project papers —
MEDAL + TABES PDFs are already on hand, the content is known, and good eval
questions with known-correct sources are easy to write. Fallback: LangGraph /
Qdrant docs pages saved as markdown.

## Day 1 — Baseline end-to-end + first numbers

- [ ] Clone repo, create venv, `pip install -e ".[dev,eval]"`, copy `.env.example` → `.env`, fill `OPENAI_API_KEY`
- [ ] `docker compose up -d qdrant` (or Qdrant Cloud free tier)
- [ ] Drop MEDAL + TABES PDFs into `data/raw/`
- [ ] Finish `ingest.py`: load → chunk (play with chunk size / overlap) → embed → upsert to Qdrant; dump `chunks.jsonl` for BM25
- [ ] Get a naive RAG path working (dense retrieve → generate) via `api.py`
- [ ] Write 10 seed eval questions in `evals/questions.jsonl` (each with expected source file)
- [ ] Run `python -m evals.run_eval` → record Run 0 row in README table

**Day 1 done = a stranger can clone, ingest, ask a question, and reproduce your baseline numbers.**

## Day 2 — Make it agentic, measure each step

- [ ] Hybrid retrieval: BM25 + dense, merged with RRF (`retriever.py`) → eval → Run 1
- [ ] Cross-encoder reranker → eval → Run 2
- [ ] LangGraph self-correction loop (`graph.py`): grade context → rewrite query → re-retrieve (max 2 retries) → eval → Run 3
- [ ] Grow question set to 30–50; add ragas metrics (faithfulness, answer relevance, context precision)
- [ ] Optional: Langfuse tracing, save 2–3 trace screenshots for README

## Day 3 — Polish into a portfolio piece

- [ ] Final eval runs; fill the README results table; write 3–5 lines on what each change bought
- [ ] 5 cited Q&A samples in README (question → answer → sources)
- [ ] Demo script or short recording
- [ ] Tidy: ruff clean, tests pass, CI green
- [ ] Write the resume bullet: "Built self-correcting Agentic RAG (LangGraph, Qdrant hybrid retrieval); raised Hit@5 from X→Y and faithfulness from A→B on a 40-question eval set"

## Stretch (only if time)

- Query routing across two collections ("stores" fallback from the original design)
- Streaming answers in the API
- Swap LLM/embedding providers via config and re-run evals
