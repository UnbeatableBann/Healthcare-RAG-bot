"""Factory for document chunking strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from core.config.settings import Settings, get_settings
from core.exceptions import ProviderNotFoundError, ProviderRegistrationError
from embeddings.base import BaseEmbedding
from rag.ingestion.chunkers.base import BaseChunker

ChunkerBuilder = Callable[[Settings, BaseEmbedding | None], BaseChunker]


class ChunkerFactory:
    """Create configured chunkers from a registry."""

    _registry: ClassVar[dict[str, ChunkerBuilder]] = {}
    _defaults_registered: ClassVar[bool] = False

    @classmethod
    def register(
        cls,
        name: str,
        builder: ChunkerBuilder,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a chunking strategy builder."""

        normalized = name.strip().lower()
        if not normalized:
            raise ProviderRegistrationError("Chunker strategy name cannot be empty.")
        if normalized in cls._registry and not overwrite:
            raise ProviderRegistrationError(
                f"Chunker strategy '{normalized}' is already registered."
            )
        cls._registry[normalized] = builder

    @classmethod
    def create(
        cls,
        strategy: str | None = None,
        *,
        settings: Settings | None = None,
        embedding: BaseEmbedding | None = None,
    ) -> BaseChunker:
        """Create the configured chunker."""

        cls.register_default_chunkers()
        resolved_settings = settings or get_settings()
        strategy_name = (
            strategy or resolved_settings.DEFAULT_CHUNKING_STRATEGY
        ).strip().lower()
        builder = cls._registry.get(strategy_name)
        if builder is None:
            raise ProviderNotFoundError(
                f"Chunker strategy '{strategy_name}' is not registered.",
                details={"registered": sorted(cls._registry)},
            )
        return builder(resolved_settings, embedding)

    @classmethod
    def register_default_chunkers(cls) -> None:
        """Register built-in chunking strategies once."""

        if cls._defaults_registered:
            return
        from rag.ingestion.chunkers.contextual import ContextualChunker
        from rag.ingestion.chunkers.hybrid import HybridChunker
        from rag.ingestion.chunkers.recursive import RecursiveChunker
        from rag.ingestion.chunkers.semantic import SemanticChunker

        cls.register("recursive", lambda settings, _embedding: RecursiveChunker(settings))
        cls.register(
            "semantic",
            lambda settings, embedding: SemanticChunker(settings, embedding),
        )
        cls.register("contextual", lambda settings, _embedding: ContextualChunker(settings))
        cls.register(
            "hybrid",
            lambda settings, embedding: HybridChunker(settings, embedding),
        )
        cls._defaults_registered = True
