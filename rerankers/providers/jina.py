"""Jina reranker provider."""

from __future__ import annotations

import httpx

from core.config.settings import Settings
from core.exceptions import ExternalServiceError
from rerankers.base import BaseReranker
from rerankers.providers.bge import BGEReranker
from schemas import TextChunk


class JinaReranker(BaseReranker):
    """Jina reranker with hosted API support and local model fallback."""

    provider_name = "jina"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.RERANKER_MODEL
        self._local_reranker: BGEReranker | None = None
        if not settings.JINA_API_KEY:
            self._local_reranker = BGEReranker(
                settings,
                provider_name=self.provider_name,
                trust_remote_code=True,
            )

    def rerank(self, query: str, chunks: list[TextChunk], top_k: int) -> list[TextChunk]:
        """Rerank chunks using Jina's hosted endpoint or local fallback."""

        if not chunks:
            return []
        if self._local_reranker is not None:
            return self._local_reranker.rerank(query, chunks, top_k)

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [chunk.content for chunk in chunks],
            "top_n": top_k,
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.JINA_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                self.settings.JINA_RERANKER_URL,
                json=payload,
                headers=headers,
                timeout=self.settings.RERANKER_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Jina reranker request failed.",
                details={"provider": self.provider_name, "error": str(exc)},
            ) from exc

        data = response.json()
        ranked: list[TextChunk] = []
        for item in data.get("results", []):
            index = int(item["index"])
            score = float(item.get("relevance_score", 0.0))
            ranked.append(chunks[index].model_copy(update={"reranker_score": score}))
        return ranked[:top_k]
