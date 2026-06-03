"""Hybrid document-aware chunking."""

from __future__ import annotations

from core.config.settings import Settings
from embeddings.base import BaseEmbedding
from rag.ingestion.chunkers.base import BaseChunker
from rag.ingestion.chunkers.contextual import ContextualChunker
from rag.ingestion.chunkers.recursive import RecursiveChunker
from rag.ingestion.chunkers.semantic import SemanticChunker
from schemas import Document, TextChunk


class HybridChunker(BaseChunker):
    """Select the best chunking strategy for each document type."""

    strategy_name = "hybrid"

    def __init__(self, settings: Settings, embedding: BaseEmbedding | None = None) -> None:
        self.recursive = RecursiveChunker(settings)
        self.semantic = SemanticChunker(settings, embedding)
        self.contextual = ContextualChunker(settings)

    def chunk(self, document: Document) -> list[TextChunk]:
        """Route a document to recursive, semantic, or contextual chunking."""

        document_type = document.metadata.document_type.lower()
        if any(key in document_type for key in ("procedure", "workflow", "instruction")):
            return self.contextual.chunk(document)
        if any(key in document_type for key in ("article", "education", "longform")):
            return self.semantic.chunk(document)
        return self.recursive.chunk(document)
