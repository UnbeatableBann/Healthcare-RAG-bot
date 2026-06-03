"""Unit tests for CRAG evaluation."""

from __future__ import annotations

import pytest

from core.config.settings import Settings
from rag.evaluation.crag_evaluator import CRAGEvaluator
from schemas import DocumentMetadata, TextChunk


def _chunk(score: float, reranker_score: float) -> TextChunk:
    return TextChunk(
        content="Patients may request medication refills through telehealth.",
        metadata=DocumentMetadata(
            document_name="telehealth.md",
            document_type="telehealth_policy",
            chunk_strategy="recursive",
            chunk_id="chunk-1",
        ),
        score=score,
        reranker_score=reranker_score,
    )


@pytest.mark.asyncio
async def test_crag_high_confidence_allows_generation(tmp_path) -> None:
    """High-confidence retrieval should proceed without a judge."""

    settings = Settings(PROJECT_ROOT=tmp_path)
    decision = await CRAGEvaluator(settings=settings).evaluate(
        "Can patients request medication refills through telehealth?",
        [_chunk(0.95, 0.95)],
    )

    assert decision.answerable is True
    assert decision.outcome == "high"


@pytest.mark.asyncio
async def test_crag_low_confidence_rejects_generation(tmp_path) -> None:
    """Low-confidence retrieval should refuse generation."""

    settings = Settings(PROJECT_ROOT=tmp_path)
    decision = await CRAGEvaluator(settings=settings).evaluate(
        "What are the dental surgery rules?",
        [_chunk(0.01, 0.01)],
    )

    assert decision.answerable is False
    assert decision.outcome == "low"
