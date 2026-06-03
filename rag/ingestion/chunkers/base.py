"""Base chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from schemas import Document, TextChunk


class BaseChunker(ABC):
    """Interface implemented by document chunking strategies."""

    strategy_name: str

    @abstractmethod
    def chunk(self, document: Document) -> list[TextChunk]:
        """Split a loaded document into searchable chunks."""
