"""Unit tests for provider factories."""

from __future__ import annotations

import pytest

from core.config.settings import Settings
from llm.base import BaseLLM, LLMResponse
from llm.factory import LLMFactory


class FakeLLM(BaseLLM):
    """Fake LLM provider for factory tests."""

    provider_name = "fake"
    model_name = "fake-model"

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(text=prompt, provider=self.provider_name, model=self.model_name)


def test_llm_factory_registers_custom_provider(tmp_path) -> None:
    """Factory registration should create custom providers."""

    LLMFactory.register("fake", lambda _settings: FakeLLM(), overwrite=True)

    provider = LLMFactory.create("fake", Settings(PROJECT_ROOT=tmp_path))

    assert provider.provider_name == "fake"


def test_llm_factory_rejects_unknown_provider(tmp_path) -> None:
    """Unknown providers should raise a domain error."""

    with pytest.raises(Exception):
        LLMFactory.create("missing", Settings(PROJECT_ROOT=tmp_path))
