"""E5 embedding provider."""

from __future__ import annotations

from core.config.settings import Settings
from embeddings.providers.bge import BGEEmbedding


class E5Embedding(BGEEmbedding):
    """Sentence-transformers embedding provider for E5 models."""

    provider_name = "e5"

    def __init__(self, settings: Settings) -> None:
        super().__init__(
            settings,
            provider_name=self.provider_name,
            query_prefix="query: ",
            document_prefix="passage: ",
        )
