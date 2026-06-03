"""Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict

from schemas import TextChunk


class ReciprocalRankFusion:
    """Fuse ranked retrieval result lists using RRF."""

    def __init__(self, k: int) -> None:
        self.k = k

    def fuse(self, ranked_lists: list[list[TextChunk]], *, top_k: int) -> list[TextChunk]:
        """Fuse ranked lists and return top chunks."""

        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, TextChunk] = {}
        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                chunk_id = chunk.metadata.chunk_id or chunk.content
                scores[chunk_id] += 1.0 / (self.k + rank)
                existing = chunks_by_id.get(chunk_id)
                if existing is None or (chunk.score or 0.0) > (existing.score or 0.0):
                    chunks_by_id[chunk_id] = chunk

        if not scores:
            return []
        max_score = max(scores.values())
        fused_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            chunks_by_id[chunk_id].model_copy(update={"score": scores[chunk_id] / max_score})
            for chunk_id in fused_ids
        ]
