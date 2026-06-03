"""Grounded response generation."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from core.exceptions import GenerationError
from core.observability.metrics import GENERATION_LATENCY_SECONDS, observe_latency
from llm.base import BaseLLM
from rag.generation.prompt_builder import PromptBuilder
from schemas import AskResponse, Citation, TextChunk


class ResponseGenerator:
    """Generate grounded answers and citations."""

    def __init__(
        self,
        llm: BaseLLM,
        *,
        prompt_builder: PromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.settings = settings or get_settings()

    async def generate(
        self,
        *,
        question: str,
        chunks: list[TextChunk],
        confidence_score: float,
        answerable: bool,
    ) -> AskResponse:
        """Generate an answer when CRAG permits it."""

        if not answerable:
            return AskResponse(
                answer=self.settings.REFUSAL_MESSAGE,
                citations=[],
                confidence_score=confidence_score,
                route="rag",
                answerable=False,
            )

        system_prompt, prompt = self.prompt_builder.build(question, chunks)
        try:
            with observe_latency(
                GENERATION_LATENCY_SECONDS,
                provider=self.llm.provider_name,
                model=self.llm.model_name,
            ):
                response = await self.llm.generate(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=self.settings.LLM_TEMPERATURE,
                    max_tokens=self.settings.LLM_MAX_TOKENS,
                )
        except Exception as exc:
            raise GenerationError(
                "Grounded answer generation failed.",
                details={"error": str(exc)},
            ) from exc

        answer = response.text.strip() or self.settings.REFUSAL_MESSAGE
        return AskResponse(
            answer=answer,
            citations=self._citations(chunks),
            confidence_score=confidence_score,
            route="rag",
            answerable=bool(response.text.strip()),
        )

    @staticmethod
    def _citations(chunks: list[TextChunk]) -> list[Citation]:
        """Build citations from final context chunks."""

        citations: list[Citation] = []
        seen: set[str] = set()
        for chunk in chunks:
            chunk_id = chunk.metadata.chunk_id or ""
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            citations.append(
                Citation(
                    document_name=chunk.metadata.document_name,
                    document_type=chunk.metadata.document_type,
                    chunk_id=chunk_id,
                    chunk_strategy=chunk.metadata.chunk_strategy or "",
                )
            )
        return citations
