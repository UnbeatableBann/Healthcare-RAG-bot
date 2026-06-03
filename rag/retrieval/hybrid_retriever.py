"""Hybrid dense and sparse retriever."""

from __future__ import annotations

from typing import Any

from core.config.settings import Settings, get_settings
from rag.query_processing.deduplicator import ChunkDeduplicator
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.rrf import ReciprocalRankFusion
from rag.retrieval.sparse_retriever import SparseRetriever
from schemas import TextChunk


class HybridRetriever:
    """Combine dense and sparse results using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf = ReciprocalRankFusion(self.settings.RRF_K)
        self.deduplicator = ChunkDeduplicator()

    def retrieve(
        self,
        queries: list[str],
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Retrieve and fuse chunks across multiple query variants."""

        ranked_lists: list[list[TextChunk]] = []
        for query in queries:
            ranked_lists.append(
                self.dense_retriever.retrieve(
                    query,
                    top_k=self.settings.DENSE_TOP_K,
                    filters=filters,
                )
            )
            ranked_lists.append(
                self.sparse_retriever.retrieve(
                    query,
                    top_k=self.settings.SPARSE_TOP_K,
                    filters=filters,
                )
            )
        fused = self.rrf.fuse(ranked_lists, top_k=self.settings.HYBRID_TOP_K)
        return self.deduplicator.deduplicate(fused)
