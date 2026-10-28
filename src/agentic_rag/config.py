"""Central settings. Everything tunable lives here and is overridable via .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM + embeddings
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "papers"

    # Chunking + retrieval knobs (change these, then re-run evals)
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8        # candidates pulled per retriever before merging
    final_k: int = 4      # chunks that actually reach the generator
    max_retries: int = 2  # self-correction loop budget

    # Optional tracing
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
