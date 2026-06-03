"""vLLM OpenAI-compatible provider."""

from __future__ import annotations

from typing import Any

import httpx

from core.config.settings import Settings
from core.exceptions import ExternalServiceError
from llm.base import BaseLLM, LLMResponse


class VLLMLLM(BaseLLM):
    """vLLM provider using the OpenAI-compatible chat completions API."""

    provider_name = "vllm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.VLLM_MODEL
        self.base_url = settings.VLLM_BASE_URL.rstrip("/")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text through vLLM chat completions."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
            if temperature is not None
            else self.settings.LLM_TEMPERATURE,
            "max_tokens": max_tokens or self.settings.LLM_MAX_TOKENS,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.settings.LLM_REQUEST_TIMEOUT_SECONDS
            ) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                "vLLM generation request failed.",
                details={"provider": self.provider_name, "error": str(exc)},
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = str(choices[0].get("message", {}).get("content", ""))
        return LLMResponse(
            text=text.strip(),
            provider=self.provider_name,
            model=self.model_name,
            raw=data,
        )
