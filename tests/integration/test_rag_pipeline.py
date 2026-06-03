"""Integration tests for the RAG pipeline with fake dependencies."""

from __future__ import annotations

import pytest

from llm.base import BaseLLM, LLMResponse
from rag.evaluation.crag_evaluator import CRAGEvaluator
from rag.generation.response_generator import ResponseGenerator
from rag.pipeline import RAGPipeline
from rag.query_processing.pipeline import QueryProcessingPipeline
from rag.reranking.rerank_service import RerankService
from rag.retrieval.hybrid_retriever import HybridRetriever
from rerankers.base import BaseReranker
from schemas import DocumentMetadata, TextChunk

pytestmark = pytest.mark.integration


class FakeLLM(BaseLLM):
    """Fake LLM for RAG integration."""

    provider_name = "fake"
    model_name = "fake"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(text="Patients may request refills online. [1]", provider="fake", model="fake")


class FakeRetriever(HybridRetriever):
    """Fake retriever returning one high-quality chunk."""

    def __init__(self) -> None:
        pass

    def retrieve(self, queries: list[str], *, filters=None) -> list[TextChunk]:
        return [
            TextChunk(
                content="Patients may request medication refills online.",
                metadata=DocumentMetadata(
                    document_name="policy.md",
                    document_type="telehealth_policy",
                    chunk_strategy="recursive",
                    chunk_id="chunk-1",
                ),
                score=1.0,
            )
        ]


class FakeReranker(BaseReranker):
    """Fake reranker."""

    provider_name = "fake"
    model_name = "fake"

    def rerank(self, query: str, chunks: list[TextChunk], top_k: int) -> list[TextChunk]:
        return [chunk.model_copy(update={"reranker_score": 1.0}) for chunk in chunks[:top_k]]


@pytest.mark.asyncio
async def test_rag_pipeline_answers_from_context() -> None:
    """Pipeline should retrieve, validate, and generate an answer."""

    llm = FakeLLM()
    pipeline = RAGPipeline(
        query_processor=QueryProcessingPipeline(llm=None),
        retriever=FakeRetriever(),
        rerank_service=RerankService(FakeReranker()),
        crag_evaluator=CRAGEvaluator(),
        response_generator=ResponseGenerator(llm),
    )

    response = await pipeline.ask("Can patients request medication refills online?")

    assert response.answerable is True
    assert response.citations
