"""Smoke tests: package imports and settings load without any external services."""


def test_settings_load():
    from agentic_rag.config import Settings

    s = Settings(openai_api_key="test")
    assert s.collection_name
    assert s.max_retries >= 0


def test_modules_import():
    import agentic_rag.api  # noqa: F401
    import agentic_rag.graph  # noqa: F401
    import agentic_rag.ingest  # noqa: F401
    import agentic_rag.retriever  # noqa: F401


def test_rrf_merge():
    from agentic_rag.retriever import reciprocal_rank_fusion

    merged = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert merged[0] in (1, 2)  # top of both lists must win
    assert set(merged) == {1, 2, 3, 4}
