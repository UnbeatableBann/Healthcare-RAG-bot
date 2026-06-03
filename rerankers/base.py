"""Base reranker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import TextChunk


class BaseReranker(ABC):
    """Interface implemented by all reranker providers."""

    provider_name: str
    model_name: str

    @abstractmethod
    def rerank(self, query: str, chunks: list[TextChunk], top_k: int) -> list[TextChunk]:
        """Return reranked chunks with reranker scores populated."""
