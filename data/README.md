# Corpus

Put source documents in `raw/` (PDF, .md, .txt). PDFs are git-ignored —
they're papers, not ours to redistribute.

Recommended starting corpus: the CSCI 544 project papers (MEDAL, TABES).
Rename them to match the eval set's `expected_source` values:

- `raw/medal.pdf`
- `raw/tabes.pdf`

Then run `python -m agentic_rag.ingest`.
