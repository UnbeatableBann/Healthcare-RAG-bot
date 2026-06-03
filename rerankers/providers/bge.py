"""BGE reranker provider."""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from core.config.settings import Settings
from rerankers.base import BaseReranker
from schemas import TextChunk


class BGEReranker(BaseReranker):
    """Cross-encoder reranker for BGE reranker models."""

    provider_name = "bge"

    def __init__(
        self,
        settings: Settings,
        *,
        provider_name: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.settings = settings
        self.provider_name = provider_name or self.provider_name
        self.model_name = settings.RERANKER_MODEL
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if settings.EMBEDDING_DEVICE:
            model_kwargs["device"] = settings.EMBEDDING_DEVICE
        self._model = CrossEncoder(self.model_name, **model_kwargs)

    def rerank(self, query: str, chunks: list[TextChunk], top_k: int) -> list[TextChunk]:
        """Score query/chunk pairs and return the top chunks."""

        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores = self._model.predict(pairs)
        scores = np.asarray(raw_scores, dtype=float)
        normalized = 1.0 / (1.0 + np.exp(-scores))
        ranked = sorted(
            zip(chunks, normalized, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            chunk.model_copy(update={"reranker_score": float(score)})
            for chunk, score in ranked[:top_k]
        ]
