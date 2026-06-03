"""Dense vector retriever."""

from __future__ import annotations

from typing import Any

from core.config.settings import Settings, get_settings
from embeddings.base import BaseEmbedding
from vectorstores.base import BaseVectorStore
from schemas import TextChunk


class DenseRetriever:
    """Retrieve semantically relevant chunks using embeddings and Qdrant."""

    def __init__(
        self,
        embedding: BaseEmbedding,
        vector_store: BaseVectorStore,
        settings: Settings | None = None,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Retrieve chunks through dense vector search."""

        query_vector = self.embedding.embed_query(query)
        return self.vector_store.search(
            query_vector,
            top_k=top_k or self.settings.DENSE_TOP_K,
            filters=filters,
        )
