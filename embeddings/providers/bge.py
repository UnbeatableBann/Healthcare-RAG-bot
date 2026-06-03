"""BGE embedding provider."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config.settings import Settings
from embeddings.base import BaseEmbedding


class BGEEmbedding(BaseEmbedding):
    """Sentence-transformers embedding provider for BGE-style models."""

    provider_name = "bge"

    def __init__(
        self,
        settings: Settings,
        *,
        provider_name: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        self.settings = settings
        self.provider_name = provider_name or self.provider_name
        self.model_name = settings.EMBEDDING_MODEL
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if settings.EMBEDDING_DEVICE:
            model_kwargs["device"] = settings.EMBEDDING_DEVICE
        self._model = SentenceTransformer(self.model_name, **model_kwargs)
        self.dimensions = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using normalized dense vectors."""

        if not texts:
            return []
        prepared = [f"{self.document_prefix}{text}" for text in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=self.settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a retrieval query."""

        vector = self._model.encode(
            [f"{self.query_prefix}{query}"],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()
