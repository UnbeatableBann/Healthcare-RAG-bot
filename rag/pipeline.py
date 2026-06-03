"""End-to-end RAG pipeline."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from core.observability.metrics import (
    HALLUCINATION_PREVENTED_TOTAL,
    RAG_QUERIES_TOTAL,
    RETRIEVAL_LATENCY_SECONDS,
    observe_latency,
)
from embeddings.factory import EmbeddingFactory
from llm.factory import LLMFactory
from rag.evaluation.crag_evaluator import CRAGEvaluator
from rag.evaluation.llm_judge import LLMJudge
from rag.generation.response_generator import ResponseGenerator
from rag.query_processing.pipeline import QueryProcessingPipeline
from rag.reranking.rerank_service import RerankService
from rag.retrieval.dense_retriever import DenseRetriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.sparse_retriever import SparseRetriever
from rerankers.factory import RerankerFactory
from schemas import AskResponse
from vectorstores.factory import VectorStoreFactory


class RAGPipeline:
    """Integrate query processing, retrieval, reranking, CRAG, and generation."""

    def __init__(
        self,
        *,
        query_processor: QueryProcessingPipeline,
        retriever: HybridRetriever,
        rerank_service: RerankService,
        crag_evaluator: CRAGEvaluator,
        response_generator: ResponseGenerator,
    ) -> None:
        self.query_processor = query_processor
        self.retriever = retriever
        self.rerank_service = rerank_service
        self.crag_evaluator = crag_evaluator
        self.response_generator = response_generator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        """Build a production RAG pipeline from configured providers."""

        resolved_settings = settings or get_settings()
        llm = LLMFactory.create(settings=resolved_settings)
        embedding = EmbeddingFactory.create(settings=resolved_settings)
        vector_store = VectorStoreFactory.create(settings=resolved_settings)
        reranker = RerankerFactory.create(settings=resolved_settings)
        dense_retriever = DenseRetriever(embedding, vector_store, resolved_settings)
        sparse_retriever = SparseRetriever(resolved_settings)
        retriever = HybridRetriever(dense_retriever, sparse_retriever, resolved_settings)
        return cls(
            query_processor=QueryProcessingPipeline(llm=llm),
            retriever=retriever,
            rerank_service=RerankService(reranker, resolved_settings),
            crag_evaluator=CRAGEvaluator(
                llm_judge=LLMJudge(llm),
                settings=resolved_settings,
            ),
            response_generator=ResponseGenerator(llm, settings=resolved_settings),
        )

    async def ask(self, question: str) -> AskResponse:
        """Answer a question with strict retrieval grounding."""

        processed_query = await self.query_processor.process(question)
        with observe_latency(RETRIEVAL_LATENCY_SECONDS, retriever="hybrid"):
            retrieved_chunks = self.retriever.retrieve(processed_query.query_variants)
        reranked_chunks = self.rerank_service.rerank(
            processed_query.rewritten_query,
            retrieved_chunks,
        )
        decision = await self.crag_evaluator.evaluate(question, reranked_chunks)
        if not decision.answerable:
            HALLUCINATION_PREVENTED_TOTAL.inc()
        response = await self.response_generator.generate(
            question=question,
            chunks=reranked_chunks,
            confidence_score=decision.confidence_score,
            answerable=decision.answerable,
        )
        RAG_QUERIES_TOTAL.labels(
            outcome="answered" if response.answerable else "refused"
        ).inc()
        return response
