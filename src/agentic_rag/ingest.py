"""Ingest documents from data/raw/ into Qdrant, and dump chunks for BM25.

Usage:  python -m agentic_rag.ingest

Pipeline: load (PDF / md / txt) -> chunk -> embed -> upsert.
Also writes data/processed/chunks.jsonl, which the BM25 side of the
hybrid retriever reads at query time.
"""

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from agentic_rag.config import settings

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_documents() -> list:
    docs = []
    for path in sorted(RAW_DIR.iterdir()):
        if path.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif path.suffix in {".md", ".txt"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def main() -> None:
    raw_docs = load_documents()
    if not raw_docs:
        raise SystemExit(f"No documents found in {RAW_DIR}/ — add PDFs/md/txt first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"{len(raw_docs)} documents -> {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    vectors = embeddings.embed_documents([c.page_content for c in chunks])

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    client.recreate_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=settings.collection_name,
        points=[
            PointStruct(
                id=i,
                vector=vec,
                payload={"text": chunk.page_content, "source": chunk.metadata.get("source", "")},
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(
                {"text": chunk.page_content, "source": chunk.metadata.get("source", "")},
                ensure_ascii=False,
            ) + "\n")
    print(f"Indexed {len(chunks)} chunks into '{settings.collection_name}'")


if __name__ == "__main__":
    main()
