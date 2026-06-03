"""BM25 sparse retriever."""

from __future__ import annotations

import math
import re
from typing import Any

from rank_bm25 import BM25Okapi

from common.utils import read_json
from core.config.settings import Settings, get_settings
from schemas import TextChunk


class SparseRetriever:
    """Keyword retriever backed by BM25 over persisted chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._chunks: list[TextChunk] = []
        self._bm25: BM25Okapi | None = None
        self._load_corpus()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Retrieve chunks using BM25."""

        self._load_corpus()
        if self._bm25 is None or not self._chunks:
            return []

        candidates = self._apply_filters(self._chunks, filters)
        if not candidates:
            return []
        candidate_indexes = [self._chunks.index(chunk) for chunk in candidates]
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            ((candidates[index], float(scores[candidate_indexes[index]])) for index in range(len(candidates))),
            key=lambda item: item[1],
            reverse=True,
        )
        positive_scores = [score for _chunk, score in ranked if score > 0.0]
        max_score = max(positive_scores) if positive_scores else 1.0
        results: list[TextChunk] = []
        for chunk, score in ranked[: top_k or self.settings.SPARSE_TOP_K]:
            normalized = max(score / max_score, 0.0) if math.isfinite(score) else 0.0
            results.append(chunk.model_copy(update={"score": normalized}))
        return results

    def _load_corpus(self) -> None:
        """Load persisted chunks into BM25."""

        payload = read_json(self.settings.PROCESSED_CHUNKS_PATH, default=[])
        chunks = [TextChunk.model_validate(item) for item in payload if isinstance(item, dict)]
        if chunks == self._chunks:
            return
        self._chunks = chunks
        tokenized = [self._tokenize(chunk.content) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25."""

        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _apply_filters(
        chunks: list[TextChunk],
        filters: dict[str, Any] | None,
    ) -> list[TextChunk]:
        """Apply metadata filters."""

        if not filters:
            return chunks
        filtered: list[TextChunk] = []
        for chunk in chunks:
            include = True
            for key, value in filters.items():
                if getattr(chunk.metadata, key, None) != value:
                    include = False
                    break
            if include:
                filtered.append(chunk)
        return filtered
