"""Factory for embedding providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core.config.settings import Settings, get_settings
from core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from embeddings.base import BaseEmbedding

EmbeddingBuilder = Callable[[Settings], BaseEmbedding]


class EmbeddingFactory:
    """Create configured embedding providers from a registry."""

    _registry: ClassVar[dict[str, EmbeddingBuilder]] = {}
    _defaults_registered: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        name: str,
        builder: EmbeddingBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register an embedding provider builder."""

        normalized = name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError("Embedding provider name cannot be empty.")
        if normalized in cls._registry and not overwrite:
            raise ProviderRegistrationError(
                f"Embedding provider '{normalized}' is already registered."
            )
        cls._registry[normalized] = builder

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        settings: Settings | None = None,
    ) -> BaseEmbedding:
        """Create the configured embedding provider."""

        cls.register_default_providers()
        resolved_settings = settings or get_settings()
        provider_name = (provider or resolved_settings.EMBEDDING_PROVIDER).strip().lower()
        builder = cls._registry.get(provider_name)
        if builder is None:
            raise ProviderNotFoundError(
                f"Embedding provider '{provider_name}' is not registered.",
                details={"registered": sorted(cls._registry)},
            )
        return builder(resolved_settings)

    @classmethod
    def register_default_providers(cls) -> None:
        """Register built-in embedding providers once."""

        if cls._defaults_registered:
            return
        from embeddings.providers.bge import BGEEmbedding
        from embeddings.providers.e5 import E5Embedding
        from embeddings.providers.nomic import NomicEmbedding

        cls.register("bge", BGEEmbedding, overwrite=True)
        cls.register("e5", E5Embedding, overwrite=True)
        cls.register("nomic", NomicEmbedding, overwrite=True)
        cls._defaults_registered = True
