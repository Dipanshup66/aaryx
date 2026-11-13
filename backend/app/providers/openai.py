"""OpenAI cloud AI provider adapter.

Communicates with OpenAI API endpoints:
- POST /v1/chat/completions (chat completions & streaming)
- GET /v1/models (model listing and health verification)
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import SecretStr

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    TokenUsage,
)

logger = get_logger("providers.openai")


class OpenAIProvider(LLMProvider):
    """Cloud LLM provider adapter connecting to OpenAI API."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize OpenAI provider with credential validation and HTTP client setup."""
        settings = get_settings()

        # Validate and extract API key securely
        resolved_key = api_key if api_key is not None else settings.openai_api_key
        if isinstance(resolved_key, SecretStr):
            raw_key = resolved_key.get_secret_value()
        elif isinstance(resolved_key, str):
            raw_key = resolved_key.strip()
        else:
            raw_key = ""

        if not raw_key:
            raise ProviderConfigurationError(
                "OpenAI API key is missing or empty. Please configure OPENAI_API_KEY.",
                provider="openai",
            )

        self._api_key = raw_key
        self.base_url = (base_url or settings.openai_base_url).strip().rstrip("/")
        self.default_model = default_model or settings.openai_model
        self.timeout = timeout if timeout is not None else settings.openai_timeout
        self.connect_timeout = connect_timeout

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(
                timeout=self.timeout,
                connect=self.connect_timeout,
            ),
        )

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Format domain messages into OpenAI's expected JSON array."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete non-streaming chat completion."""
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._format_messages(request.messages),
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        logger.info("Dispatching OpenAI completion request (model=%s)", target_model)

        try:
            response = await self._client.post(endpoint, json=payload, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            logger.error("OpenAI request timed out after %.1fs", self.timeout)
            raise ProviderTimeoutError(
                f"OpenAI request timed out after {self.timeout}s", provider="openai"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to OpenAI at %s", self.base_url)
            raise ProviderConnectionError(
                f"Cannot connect to OpenAI service at {self.base_url}", provider="openai"
            ) from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            text = exc.response.text
            logger.error("OpenAI returned HTTP error status %d", code)
            if code in (401, 403):
                raise ProviderResponseError(
                    f"OpenAI authentication error ({code}): {text}", provider="openai"
                ) from exc
            if code == 429:
                raise ProviderResponseError(
                    f"OpenAI rate limit or quota exceeded ({code}): {text}", provider="openai"
                ) from exc
            raise ProviderResponseError(
                f"OpenAI returned HTTP {code}: {text}", provider="openai"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("Failed to parse OpenAI response: %s", exc)
            raise ProviderResponseError(
                f"Invalid response from OpenAI: {exc}", provider="openai"
            ) from exc

        choices = data.get("choices", [])
        if not choices:
            raise ProviderResponseError("OpenAI response contained no choices", provider="openai")

        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")
        model_used = data.get("model", target_model)

        usage: TokenUsage | None = None
        raw_usage = data.get("usage")
        if raw_usage and isinstance(raw_usage, dict):
            usage = TokenUsage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
            )

        logger.info("OpenAI completion succeeded (model=%s)", model_used)
        return ChatResponse(
            content=content,
            model=model_used,
            finish_reason=finish_reason,
            usage=usage,
            provider="openai",
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks asynchronously via SSE."""
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._format_messages(request.messages),
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        logger.info("Opening OpenAI streaming stream (model=%s)", target_model)

        try:
            async with self._client.stream(
                "POST", endpoint, json=payload, headers=self._headers
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    code = response.status_code
                    text = error_body.decode(errors="replace")
                    logger.error("OpenAI streaming returned HTTP %d: %s", code, text)
                    if code in (401, 403):
                        raise ProviderResponseError(
                            f"OpenAI authentication error ({code}): {text}", provider="openai"
                        )
                    if code == 429:
                        raise ProviderResponseError(
                            f"OpenAI rate limit exceeded ({code}): {text}", provider="openai"
                        )
                    raise ProviderResponseError(
                        f"OpenAI streaming error ({code}): {text}", provider="openai"
                    )

                async for line in response.aiter_lines():
                    cleaned_line = line.strip()
                    if not cleaned_line or not cleaned_line.startswith("data:"):
                        continue

                    data_str = cleaned_line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed OpenAI stream frame: %s", exc)
                        continue

                    choices = chunk_json.get("choices", [])
                    if not choices:
                        continue

                    delta_text = choices[0].get("delta", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason")
                    chunk_model = chunk_json.get("model", target_model)

                    if not delta_text and finish_reason is None:
                        continue

                    yield ChatStreamChunk(
                        delta=delta_text,
                        model=chunk_model,
                        finish_reason=finish_reason,
                        provider="openai",
                    )
        except httpx.TimeoutException as exc:
            logger.error("OpenAI stream timed out")
            raise ProviderTimeoutError(
                f"OpenAI stream timed out after {self.timeout}s", provider="openai"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("OpenAI stream network connection failed")
            raise ProviderConnectionError(
                "OpenAI stream connection failed", provider="openai"
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from OpenAI."""
        endpoint = f"{self.base_url}/models"
        try:
            response = await self._client.get(endpoint, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"OpenAI models list timed out: {exc}", provider="openai"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderConnectionError(
                f"Cannot connect to OpenAI at {self.base_url}: {exc}", provider="openai"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Failed to retrieve OpenAI models: {exc}", provider="openai"
            ) from exc

        model_items = data.get("data", [])
        return [
            ModelInfo(
                name=m.get("id", "unknown"),
                modified_at=str(m.get("created")) if m.get("created") else None,
            )
            for m in model_items
            if isinstance(m, dict) and "id" in m
        ]

    async def health_check(self) -> ProviderHealth:
        """Verify OpenAI connectivity and key validity."""
        try:
            models = await self.list_models()
            return ProviderHealth(
                healthy=True,
                provider="openai",
                details={
                    "base_url": self.base_url,
                    "default_model": self.default_model,
                    "model_count": len(models),
                },
            )
        except ProviderError as exc:
            logger.warning("OpenAI health check failed: %s", exc)
            return ProviderHealth(
                healthy=False,
                provider="openai",
                details={"base_url": self.base_url, "default_model": self.default_model},
                error=str(exc),
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._own_client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed OpenAI HTTP client")
