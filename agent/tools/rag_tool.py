"""RAG tool used by the healthcare agent."""

from __future__ import annotations

from rag.pipeline import RAGPipeline
from schemas import AskResponse


class RAGTool:
    """Agent tool wrapper around the RAG pipeline."""

    name = "rag_tool"

    def __init__(self, pipeline: RAGPipeline) -> None:
        self.pipeline = pipeline

    async def run(self, question: str) -> AskResponse:
        """Answer a knowledge-base question."""

        return await self.pipeline.ask(question)
