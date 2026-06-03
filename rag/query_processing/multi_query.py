"""Multi-query generation."""

from __future__ import annotations

import re

from core.config.settings import Settings, get_settings
from core.logging.logger import get_logger
from llm.base import BaseLLM

logger = get_logger(__name__)


class MultiQueryGenerator:
    """Generate semantically related retrieval query variants."""

    def __init__(self, llm: BaseLLM | None = None, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    async def generate(self, original_query: str, rewritten_query: str) -> list[str]:
        """Generate query variants, always including the rewritten query."""

        variants = [rewritten_query]
        if self.llm is not None and self.settings.MULTI_QUERY_COUNT > 1:
            prompt = (
                "Generate concise healthcare document retrieval queries. "
                f"Return exactly {self.settings.MULTI_QUERY_COUNT} lines and no commentary.\n\n"
                f"Original question: {original_query}\n"
                f"Retrieval query: {rewritten_query}"
            )
            try:
                response = await self.llm.generate(
                    prompt,
                    temperature=0.2,
                    max_tokens=256,
                )
                variants.extend(self._parse_lines(response.text))
            except Exception as exc:
                logger.warning("Multi-query generation failed; using fallback", error=str(exc))
                variants.extend(self._fallback_variants(rewritten_query))
        else:
            variants.extend(self._fallback_variants(rewritten_query))

        return self._deduplicate(variants)[: self.settings.MULTI_QUERY_COUNT]

    @staticmethod
    def _parse_lines(text: str) -> list[str]:
        """Parse line-oriented LLM output."""

        lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)
        return lines

    @staticmethod
    def _fallback_variants(query: str) -> list[str]:
        """Generate deterministic query variants."""

        return [
            query.replace("patient", "member"),
            query.replace("policy", "guideline"),
            query.replace("medication", "medicine"),
        ]

    @staticmethod
    def _deduplicate(queries: list[str]) -> list[str]:
        """Remove duplicate queries while preserving order."""

        seen: set[str] = set()
        unique: list[str] = []
        for query in queries:
            normalized = query.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique
