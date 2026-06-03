"""Reranking orchestration."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from core.observability.metrics import RERANKING_LATENCY_SECONDS, observe_latency
from rerankers.base import BaseReranker
from schemas import TextChunk


class RerankService:
    """Apply cross-encoder reranking to hybrid retrieval candidates."""

    def __init__(self, reranker: BaseReranker, settings: Settings | None = None) -> None:
        self.reranker = reranker
        self.settings = settings or get_settings()

    def rerank(self, query: str, chunks: list[TextChunk]) -> list[TextChunk]:
        """Rerank top 30, retain top 10, and return final top 5."""

        if not chunks:
            return []
        candidates = chunks[: self.settings.HYBRID_TOP_K]
        with observe_latency(
            RERANKING_LATENCY_SECONDS,
            reranker=self.reranker.provider_name,
        ):
            top_reranked = self.reranker.rerank(
                query,
                candidates,
                top_k=self.settings.RERANKER_TOP_N,
            )
        return top_reranked[: self.settings.FINAL_CONTEXT_TOP_K]
