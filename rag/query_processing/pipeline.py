"""Query processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from llm.base import BaseLLM
from rag.query_processing.multi_query import MultiQueryGenerator
from rag.query_processing.rewriter import QueryRewriter


@dataclass(frozen=True)
class ProcessedQuery:
    """Output of query processing."""

    original_query: str
    rewritten_query: str
    query_variants: list[str]


class QueryProcessingPipeline:
    """Rewrite a question and generate multi-query retrieval variants."""

    def __init__(
        self,
        *,
        rewriter: QueryRewriter | None = None,
        multi_query_generator: MultiQueryGenerator | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self.rewriter = rewriter or QueryRewriter(llm)
        self.multi_query_generator = multi_query_generator or MultiQueryGenerator(llm)

    async def process(self, question: str) -> ProcessedQuery:
        """Process a user question for retrieval."""

        rewritten = await self.rewriter.rewrite(question)
        variants = await self.multi_query_generator.generate(question, rewritten)
        return ProcessedQuery(
            original_query=question,
            rewritten_query=rewritten,
            query_variants=variants,
        )
