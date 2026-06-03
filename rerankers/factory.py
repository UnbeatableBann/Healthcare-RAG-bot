"""Factory for reranker providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core.config.settings import Settings, get_settings
from core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from rerankers.base import BaseReranker

RerankerBuilder = Callable[[Settings], BaseReranker]


class RerankerFactory:
    """Create configured reranker providers from a registry."""

    _registry: ClassVar[dict[str, RerankerBuilder]] = {}
    _defaults_registered: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        name: str,
        builder: RerankerBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a reranker provider builder."""

        normalized = name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError("Reranker provider name cannot be empty.")
        if normalized in cls._registry and not overwrite:
            raise ProviderRegistrationError(
                f"Reranker provider '{normalized}' is already registered."
            )
        cls._registry[normalized] = builder

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        settings: Settings | None = None,
    ) -> BaseReranker:
        """Create the configured reranker provider."""

        cls.register_default_providers()
        resolved_settings = settings or get_settings()
        provider_name = (provider or resolved_settings.RERANKER_PROVIDER).strip().lower()
        builder = cls._registry.get(provider_name)
        if builder is None:
            raise ProviderNotFoundError(
                f"Reranker provider '{provider_name}' is not registered.",
                details={"registered": sorted(cls._registry)},
            )
        return builder(resolved_settings)

    @classmethod
    def register_default_providers(cls) -> None:
        """Register built-in reranker providers once."""

        if cls._defaults_registered:
            return
        from rerankers.providers.bge import BGEReranker
        from rerankers.providers.jina import JinaReranker

        cls.register("bge", BGEReranker, overwrite=True)
        cls.register("jina", JinaReranker, overwrite=True)
        cls._defaults_registered = True
