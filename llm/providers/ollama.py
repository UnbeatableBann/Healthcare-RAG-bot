"""Ollama LLM provider."""

from __future__ import annotations

from typing import Any

import httpx

from core.config.settings import Settings
from core.exceptions import ExternalServiceError
from llm.base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """Ollama HTTP API implementation."""

    provider_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text through Ollama `/api/generate`."""

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
                if temperature is not None
                else self.settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or self.settings.LLM_MAX_TOKENS,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.LLM_REQUEST_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "Ollama generation request failed.",
                details={"provider": self.provider_name, "error": str(exc)},
            ) from exc

        data = response.json()
        return LLMResponse(
            text=str(data.get("response", "")).strip(),
            provider=self.provider_name,
            model=self.model_name,
            raw=data,
        )
