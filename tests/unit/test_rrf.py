"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

from rag.retrieval.rrf import ReciprocalRankFusion
from schemas import DocumentMetadata, TextChunk


def _chunk(chunk_id: str, score: float) -> TextChunk:
    return TextChunk(
        content=f"content {chunk_id}",
        metadata=DocumentMetadata(
            document_name="doc.md",
            document_type="policy",
            chunk_strategy="recursive",
            chunk_id=chunk_id,
        ),
        score=score,
    )


def test_rrf_fuses_and_deduplicates_ranked_lists() -> None:
    """RRF should combine duplicate IDs and normalize scores."""

    fused = ReciprocalRankFusion(k=60).fuse(
        [[_chunk("a", 0.8), _chunk("b", 0.7)], [_chunk("b", 0.9), _chunk("c", 0.5)]],
        top_k=3,
    )

    assert [chunk.metadata.chunk_id for chunk in fused] == ["b", "a", "c"]
    assert fused[0].score == 1.0
