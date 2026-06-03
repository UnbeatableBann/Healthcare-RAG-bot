"""RAGAS evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RAGASSample:
    """Single RAGAS evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


class RAGASRunner:
    """Run RAGAS metrics for RAG quality evaluation."""

    metric_names = (
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    )

    def evaluate(self, samples: list[RAGASSample]) -> dict[str, float]:
        """Evaluate RAGAS metrics for samples."""

        if not samples:
            return {name: 0.0 for name in self.metric_names}

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_list(
            [
                {
                    "question": sample.question,
                    "answer": sample.answer,
                    "contexts": sample.contexts,
                    "ground_truth": sample.ground_truth,
                }
                for sample in samples
            ]
        )
        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, float]:
        """Normalize RAGAS result objects into plain metrics."""

        if hasattr(result, "to_pandas"):
            frame = result.to_pandas()
            return {
                column: float(frame[column].mean())
                for column in frame.columns
                if column
                in {
                    "context_precision",
                    "context_recall",
                    "faithfulness",
                    "answer_relevancy",
                }
            }
        payload = dict(result)
        return {key: float(value) for key, value in payload.items()}
