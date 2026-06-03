"""LLM-as-a-judge answerability validation."""

from __future__ import annotations

from llm.base import BaseLLM
from schemas import TextChunk


class LLMJudge:
    """Judge whether retrieved context can answer a question."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def is_answerable(self, question: str, chunks: list[TextChunk]) -> bool:
        """Return True only when the LLM judge outputs ANSWERABLE."""

        context = "\n\n".join(chunk.content for chunk in chunks)
        prompt = (
            "Decide whether the retrieved healthcare context fully supports "
            "answering the question. Output exactly one token: ANSWERABLE or "
            "NOT_ANSWERABLE.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}"
        )
        response = await self.llm.generate(prompt, temperature=0.0, max_tokens=8)
        verdict = response.text.strip().upper()
        if "NOT_ANSWERABLE" in verdict:
            return False
        return "ANSWERABLE" in verdict
