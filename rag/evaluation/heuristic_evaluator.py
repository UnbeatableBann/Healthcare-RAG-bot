"""Heuristic retrieval quality evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.config.settings import Settings, get_settings
from schemas import TextChunk


@dataclass(frozen=True)
class HeuristicEvaluation:
    """Heuristic CRAG evaluation result."""

    confidence_score: float
    outcome: str
    retrieval_score: float
    reranker_score: float
    context_coverage_score: float


class HeuristicEvaluator:
    """Evaluate retrieval quality using score and coverage signals."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def evaluate(self, question: str, chunks: list[TextChunk]) -> HeuristicEvaluation:
        """Return high, medium, or low confidence for retrieved context."""

        retrieval_score = self._average_score(chunk.score for chunk in chunks)
        reranker_score = self._average_score(chunk.reranker_score for chunk in chunks)
        coverage_score = self._context_coverage(question, chunks)
        weight_sum = (
            self.settings.CRAG_RETRIEVAL_SCORE_WEIGHT
            + self.settings.CRAG_RERANKER_SCORE_WEIGHT
            + self.settings.CRAG_CONTEXT_COVERAGE_WEIGHT
        )
        confidence = (
            retrieval_score * self.settings.CRAG_RETRIEVAL_SCORE_WEIGHT
            + reranker_score * self.settings.CRAG_RERANKER_SCORE_WEIGHT
            + coverage_score * self.settings.CRAG_CONTEXT_COVERAGE_WEIGHT
        ) / weight_sum
        confidence = max(0.0, min(confidence, 1.0))
        if confidence >= self.settings.CRAG_HIGH_CONFIDENCE_THRESHOLD:
            outcome = "high"
        elif confidence < self.settings.CRAG_LOW_CONFIDENCE_THRESHOLD:
            outcome = "low"
        else:
            outcome = "medium"
        return HeuristicEvaluation(
            confidence_score=confidence,
            outcome=outcome,
            retrieval_score=retrieval_score,
            reranker_score=reranker_score,
            context_coverage_score=coverage_score,
        )

    @staticmethod
    def _average_score(values: object) -> float:
        """Average normalized scores, ignoring missing values."""

        numbers = [float(value) for value in values if value is not None]
        if not numbers:
            return 0.0
        return max(0.0, min(sum(numbers) / len(numbers), 1.0))

    @classmethod
    def _context_coverage(cls, question: str, chunks: list[TextChunk]) -> float:
        """Measure question-token coverage in retrieved context."""

        question_terms = cls._content_terms(question)
        if not question_terms:
            return 0.0
        context_terms = cls._content_terms(" ".join(chunk.content for chunk in chunks))
        return len(question_terms & context_terms) / len(question_terms)

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        """Extract meaningful terms."""

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "can",
            "do",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
            "when",
            "where",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in stopwords
        }
