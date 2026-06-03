"""Factory for vector store providers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core.config.settings import Settings, get_settings
from core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from vectorstores.base import BaseVectorStore

VectorStoreBuilder = Callable[[Settings], BaseVectorStore]


class VectorStoreFactory:
    """Create configured vector stores from a registry."""

    _registry: ClassVar[dict[str, VectorStoreBuilder]] = {}
    _defaults_registered: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        name: str,
        builder: VectorStoreBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a vector store provider builder."""

        normalized = name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError("Vector store provider name cannot be empty.")
        if normalized in cls._registry and not overwrite:
            raise ProviderRegistrationError(
                f"Vector store provider '{normalized}' is already registered."
            )
        cls._registry[normalized] = builder

    @classmethod
    def create(
        cls,
        provider: str | None = None,
        settings: Settings | None = None,
    ) -> BaseVectorStore:
        """Create the configured vector store provider."""

        cls.register_default_providers()
        resolved_settings = settings or get_settings()
        provider_name = (provider or resolved_settings.VECTORSTORE_PROVIDER).strip().lower()
        builder = cls._registry.get(provider_name)
        if builder is None:
            raise ProviderNotFoundError(
                f"Vector store provider '{provider_name}' is not registered.",
                details={"registered": sorted(cls._registry)},
            )
        return builder(resolved_settings)

    @classmethod
    def register_default_providers(cls) -> None:
        """Register built-in vector store providers once."""

        if cls._defaults_registered:
            return
        from vectorstores.providers.qdrant import QdrantVectorStore

        cls.register("qdrant", QdrantVectorStore, overwrite=True)
        cls._defaults_registered = True
