"""Optional integration tests for external Qdrant and Ollama services."""

from __future__ import annotations

import os

import pytest

from llm.factory import LLMFactory
from vectorstores.factory import VectorStoreFactory

pytestmark = pytest.mark.integration


def _external_enabled() -> bool:
    return os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS", "").lower() == "true"


@pytest.mark.skipif(not _external_enabled(), reason="External integration tests disabled")
def test_qdrant_collection_can_be_ensured() -> None:
    """Qdrant provider should create or find its collection."""

    vector_store = VectorStoreFactory.create("qdrant")
    vector_store.ensure_collection()


@pytest.mark.asyncio
@pytest.mark.skipif(not _external_enabled(), reason="External integration tests disabled")
async def test_ollama_can_generate_text() -> None:
    """Ollama provider should return text from the configured model."""

    llm = LLMFactory.create("ollama")
    response = await llm.generate("Reply with the word READY.", max_tokens=8)

    assert response.text
