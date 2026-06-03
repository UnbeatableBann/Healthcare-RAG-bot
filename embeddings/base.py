"""Base embedding interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Interface implemented by embedding providers."""

    provider_name: str
    model_name: str
    dimensions: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a retrieval query."""

        return self.embed_texts([query])[0]
