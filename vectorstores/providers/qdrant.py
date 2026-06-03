"""Qdrant vector store implementation."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config.settings import Settings
from core.exceptions import ExternalServiceError
from schemas import DocumentMetadata, TextChunk
from vectorstores.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Dense-vector storage backed by Qdrant."""

    provider_name = "qdrant"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
        )

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if needed."""

        try:
            exists = self._collection_exists()
            if exists:
                return
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.settings.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.document_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to ensure Qdrant collection.",
                details={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        """Store chunks and vectors in Qdrant."""

        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts must match.")
        if not chunks:
            return

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.metadata.chunk_id or chunk.content)),
                vector=vector,
                payload={
                    "content": chunk.content,
                    "metadata": chunk.metadata.model_dump(mode="json"),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:
            raise ExternalServiceError(
                "Failed to upsert vectors into Qdrant.",
                details={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Search Qdrant using dense vector similarity."""

        query_filter = self._build_filter(filters)
        try:
            if hasattr(self._client, "search"):
                hits = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                )
            else:
                hits = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    with_payload=True,
                ).points
        except Exception as exc:
            raise ExternalServiceError(
                "Qdrant dense search failed.",
                details={"collection": self.collection_name, "error": str(exc)},
            ) from exc

        chunks: list[TextChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            metadata = DocumentMetadata.model_validate(payload.get("metadata", {}))
            chunks.append(
                TextChunk(
                    content=str(payload.get("content", "")),
                    metadata=metadata,
                    score=max(float(hit.score or 0.0), 0.0),
                )
            )
        return chunks

    def _collection_exists(self) -> bool:
        """Return whether the configured collection exists."""

        if hasattr(self._client, "collection_exists"):
            return bool(self._client.collection_exists(self.collection_name))
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            return False
        return True

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        """Build a Qdrant metadata filter."""

        if not filters:
            return None
        conditions = [
            models.FieldCondition(
                key=f"metadata.{key}",
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
            if value is not None
        ]
        if not conditions:
            return None
        return models.Filter(must=conditions)
