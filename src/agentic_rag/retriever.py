"""Hybrid retrieval: Qdrant dense search + BM25, merged with RRF, then reranked.

Each stage is a separate method on purpose — the eval harness measures the
pipeline with stages toggled, which is what fills the README results table.
"""

import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from agentic_rag.config import settings

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge ranked id lists into one ordering (standard RRF)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=settings.openai_api_key
        )
        self.chunks = self._load_chunks()
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])
        self._reranker = None  # lazy: cross-encoder is heavy to load

    def _load_chunks(self) -> list[dict]:
        if not CHUNKS_PATH.exists():
            raise SystemExit("data/processed/chunks.jsonl missing — run ingest first.")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # --- stage 1a: dense retrieval -------------------------------------
    def dense_search(self, query: str, limit: int) -> list[int]:
        vector = self.embeddings.embed_query(query)
        hits = self.client.search(
            collection_name=settings.collection_name, query_vector=vector, limit=limit
        )
        return [hit.id for hit in hits]

    # --- stage 1b: sparse retrieval ------------------------------------
    def bm25_search(self, query: str, limit: int) -> list[int]:
        scores = self.bm25.get_scores(query.lower().split())
        return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]

    # --- stage 2: rerank (optional) -------------------------------------
    def rerank(self, query: str, doc_ids: list[int], limit: int) -> list[int]:
        # TODO(day2): load CrossEncoder("BAAI/bge-reranker-base") lazily,
        # score (query, chunk_text) pairs, return the top `limit` ids.
        return doc_ids[:limit]

    def retrieve(
        self, query: str, use_hybrid: bool = True, use_rerank: bool = False
    ) -> list[dict]:
        """Return up to settings.final_k chunk dicts: {text, source}."""
        if use_hybrid:
            merged = reciprocal_rank_fusion([
                self.dense_search(query, settings.top_k),
                self.bm25_search(query, settings.top_k),
            ])
        else:
            merged = self.dense_search(query, settings.top_k)

        if use_rerank:
            merged = self.rerank(query, merged, settings.final_k)
        return [self.chunks[i] for i in merged[: settings.final_k]]
