"""llama.cpp server provider."""

from __future__ import annotations

from typing import Any

import httpx

from core.config.settings import Settings
from core.exceptions import ExternalServiceError
from llm.base import BaseLLM, LLMResponse


class LlamaCppLLM(BaseLLM):
    """llama.cpp HTTP server implementation."""

    provider_name = "llamacpp"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.LLAMACPP_MODEL
        self.base_url = settings.LLAMACPP_BASE_URL.rstrip("/")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text through llama.cpp `/completion`."""

        full_prompt = prompt if system_prompt is None else f"{system_prompt}\n\n{prompt}"
        payload: dict[str, Any] = {
            "prompt": full_prompt,
            "temperature": temperature
            if temperature is not None
            else self.settings.LLM_TEMPERATURE,
            "n_predict": max_tokens or self.settings.LLM_MAX_TOKENS,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.LLM_REQUEST_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(f"{self.base_url}/completion", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "llama.cpp generation request failed.",
                details={"provider": self.provider_name, "error": str(exc)},
            ) from exc

        data = response.json()
        return LLMResponse(
            text=str(data.get("content", "")).strip(),
            provider=self.provider_name,
            model=self.model_name,
            raw=data,
        )
