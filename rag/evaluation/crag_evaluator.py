"""CRAG-inspired retrieval validation."""

from __future__ import annotations

from dataclasses import dataclass

from core.config.settings import Settings, get_settings
from rag.evaluation.heuristic_evaluator import HeuristicEvaluator
from rag.evaluation.llm_judge import LLMJudge
from schemas import TextChunk


@dataclass(frozen=True)
class CRAGDecision:
    """Decision about whether generation may proceed."""

    answerable: bool
    confidence_score: float
    outcome: str
    reason: str


class CRAGEvaluator:
    """Two-stage CRAG-inspired validation."""

    def __init__(
        self,
        *,
        heuristic_evaluator: HeuristicEvaluator | None = None,
        llm_judge: LLMJudge | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.heuristic_evaluator = heuristic_evaluator or HeuristicEvaluator(self.settings)
        self.llm_judge = llm_judge

    async def evaluate(self, question: str, chunks: list[TextChunk]) -> CRAGDecision:
        """Determine whether retrieved chunks support answer generation."""

        if not chunks:
            return CRAGDecision(False, 0.0, "low", "No retrieved context.")

        heuristic = self.heuristic_evaluator.evaluate(question, chunks)
        if heuristic.outcome == "high":
            return CRAGDecision(
                True,
                heuristic.confidence_score,
                "high",
                "Heuristic confidence is high.",
            )
        if heuristic.outcome == "low":
            return CRAGDecision(
                False,
                heuristic.confidence_score,
                "low",
                "Heuristic confidence is low.",
            )
        if not self.settings.LLM_JUDGE_ENABLED or self.llm_judge is None:
            return CRAGDecision(
                False,
                heuristic.confidence_score,
                "medium",
                "Medium confidence and LLM judge unavailable.",
            )
        answerable = await self.llm_judge.is_answerable(question, chunks)
        return CRAGDecision(
            answerable,
            heuristic.confidence_score,
            "medium",
            "LLM judge marked context as answerable."
            if answerable
            else "LLM judge marked context as not answerable.",
        )
