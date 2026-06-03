"""Nomic embedding provider."""

from __future__ import annotations

from core.config.settings import Settings
from embeddings.providers.bge import BGEEmbedding


class NomicEmbedding(BGEEmbedding):
    """Sentence-transformers embedding provider for Nomic embedding models."""

    provider_name = "nomic"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            provider_name=self.provider_name,
            query_prefix="search_query: ",
            document_prefix="search_document: ",
            trust_remote_code=True,
        )
