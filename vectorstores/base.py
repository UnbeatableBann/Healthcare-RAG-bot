"""Base vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas import TextChunk


class BaseVectorStore(ABC):
    """Interface implemented by vector store providers."""

    provider_name: str

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the backing collection if it does not exist."""

    @abstractmethod
    def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        """Store chunks with dense vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Search for chunks by dense vector similarity."""
