"""Factory for LLM providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core.config.settings import Settings, get_settings
from core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from llm.base import BaseLLM

LLMBuilder = Callable[[Settings], BaseLLM]


class LLMFactory:
    """Create configured LLM providers from a registry."""

    _registry: ClassVar[dict[str, LLMBuilder]] = {}
    _defaults_registered: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        name: str,
        builder: LLMBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an LLM provider builder."""

        normalized = name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError("LLM provider name cannot be empty.")
        if normalized in cls._registry and not overwrite:
            raise ProviderRegistrationError(
                f"LLM provider '{normalized}' is already registered."
            )
        cls._registry[normalized] = builder

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        settings: Settings | None = None,
    ) -> BaseLLM:
        """Create the configured LLM provider."""

        cls.register_default_providers()
        resolved_settings = settings or get_settings()
        provider_name = (provider or resolved_settings.LLM_PROVIDER).strip().lower()
        builder = cls._registry.get(provider_name)
        if builder is None:
            raise ProviderNotFoundError(
                f"LLM provider '{provider_name}' is not registered.",
                details={"registered": sorted(cls._registry)},
            )
        return builder(resolved_settings)

    @classmethod
    def register_default_providers(cls) -> None:
        """Register built-in LLM providers once."""

        if cls._defaults_registered:
            return
        from llm.providers.llamacpp import LlamaCppLLM
        from llm.providers.ollama import OllamaLLM
        from llm.providers.vllm import VLLMLLM

        cls.register("ollama", OllamaLLM, overwrite=True)
        cls.register("vllm", VLLMLLM, overwrite=True)
        cls.register("llamacpp", LlamaCppLLM, overwrite=True)
        cls._defaults_registered = True
