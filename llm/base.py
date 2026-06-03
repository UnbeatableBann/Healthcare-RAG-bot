"""Base LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response returned by all providers."""

    text: str
    provider: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """Interface implemented by all LLM providers."""

    provider_name: str
    model_name: str

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text for a prompt."""
