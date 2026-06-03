"""Query rewriting for retrieval optimization."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from core.logging.logger import get_logger
from llm.base import BaseLLM

logger = get_logger(__name__)


class QueryRewriter:
    """Rewrite user questions into retrieval-optimized healthcare queries."""

    def __init__(self, llm: BaseLLM | None = None, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    async def rewrite(self, question: str) -> str:
        """Return a retrieval-optimized query."""

        if not self.settings.QUERY_REWRITE_ENABLED or self.llm is None:
            return self._deterministic_rewrite(question)

        prompt = (
            "Rewrite the healthcare question for document retrieval. "
            "Use formal healthcare terminology, preserve the original intent, "
            "and return only the rewritten query.\n\n"
            f"Question: {question}"
        )
        try:
            response = await self.llm.generate(
                prompt,
                temperature=0.0,
                max_tokens=128,
            )
            rewritten = response.text.strip().strip('"')
            return rewritten or self._deterministic_rewrite(question)
        except Exception as exc:
            logger.warning("Query rewrite failed; using deterministic fallback", error=str(exc))
            return self._deterministic_rewrite(question)

    @staticmethod
    def _deterministic_rewrite(question: str) -> str:
        """Apply lightweight healthcare terminology normalization."""

        rewritten = question.strip()
        replacements = {
            "refill medicine": "request medication refill",
            "refill medication": "request medication refill",
            "online doctor": "telehealth consultation",
            "video visit": "telehealth consultation",
            "insurance": "health insurance coverage",
            "privacy": "patient privacy and HIPAA",
        }
        lowered = rewritten.lower()
        for source, target in replacements.items():
            if source in lowered:
                lowered = lowered.replace(source, target)
        return lowered if lowered != question.lower() else rewritten
