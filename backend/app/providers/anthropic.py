"""Anthropic Claude cloud AI provider adapter.

Communicates with Anthropic Claude API endpoints:
- POST /v1/messages (chat completions & streaming)
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
    describe_error,
    safe_url,
)

logger = get_logger("providers.anthropic")


# Anthropic error `type` values -> equivalent HTTP status. In-stream `error` events carry no
# HTTP status (the response is already 200), so the status is derived from the documented type.
_ERROR_TYPE_STATUS: dict[str, int] = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def _error_event_to_exception(event: dict[str, Any]) -> ProviderResponseError:
    """Convert an Anthropic in-stream ``error`` event into a structured ProviderResponseError."""
    err = event.get("error")
    err = err if isinstance(err, dict) else {}
    error_type = str(err.get("type", "unknown_error"))
    message = str(err.get("message", "no message"))
    return ProviderResponseError(
        f"Anthropic stream error ({error_type}): {message}",
        provider="anthropic",
        status_code=_ERROR_TYPE_STATUS.get(error_type),
    )


class AnthropicProvider(LLMProvider):
    """Cloud LLM provider adapter connecting to Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Anthropic provider with credential validation and HTTP client setup."""
        settings = get_settings()

        resolved_key = api_key if api_key is not None else settings.anthropic_api_key
        if isinstance(resolved_key, SecretStr):
            raw_key = resolved_key.get_secret_value()
        elif isinstance(resolved_key, str):
            raw_key = resolved_key.strip()
        else:
            raw_key = ""

        if not raw_key:
            raise ProviderConfigurationError(
                "Anthropic API key is missing or empty. Please configure ANTHROPIC_API_KEY.",
                provider="anthropic",
            )

        self._api_key = raw_key
        self.base_url = (base_url or settings.anthropic_base_url).strip().rstrip("/")
        self.default_model = default_model or settings.anthropic_model
        self.timeout = timeout if timeout is not None else settings.anthropic_timeout
        self.connect_timeout = connect_timeout

        self._headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(
                timeout=self.timeout,
                connect=self.connect_timeout,
            ),
        )

    def _split_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Separate system messages from conversational turns (Anthropic API requirement)."""
        system_prompts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_prompts.append(msg.content)
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        system_str = "\n\n".join(system_prompts) if system_prompts else None
        return system_str, conversation

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete non-streaming chat completion from Claude."""
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/messages"

        system_prompt, messages = self._split_messages(request.messages)

        # max_tokens is required by Anthropic Messages API
        max_tokens = request.max_tokens or 4096

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        logger.info("Dispatching Anthropic completion request (model=%s)", target_model)

        try:
            response = await self._client.post(endpoint, json=payload, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            logger.error("Anthropic request timed out after %.1fs", self.timeout)
            raise ProviderTimeoutError(
                f"Anthropic request timed out after {self.timeout}s", provider="anthropic"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to Anthropic at %s", safe_url(self.base_url))
            raise ProviderConnectionError(
                f"Cannot connect to Anthropic service at {safe_url(self.base_url)}",
                provider="anthropic",
            ) from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            text = exc.response.text
            logger.error("Anthropic returned HTTP error status %d", code)
            if code in (401, 403):
                raise ProviderResponseError(
                    f"Anthropic authentication error ({code}): {text}",
                    provider="anthropic",
                    status_code=code,
                ) from exc
            if code == 429:
                raise ProviderResponseError(
                    f"Anthropic rate limit exceeded ({code}): {text}",
                    provider="anthropic",
                    status_code=code,
                ) from exc
            raise ProviderResponseError(
                f"Anthropic returned HTTP {code}: {text}", provider="anthropic", status_code=code
            ) from exc
        except httpx.UnsupportedProtocol as exc:
            raise ProviderConfigurationError(
                f"Invalid Anthropic base URL: {safe_url(self.base_url)}", provider="anthropic"
            ) from exc
        except httpx.TransportError as exc:
            # RemoteProtocolError, ProxyError, ... : the connection failed or dropped.
            logger.error("Anthropic transport failure: %s", exc.__class__.__name__)
            raise ProviderConnectionError(
                f"Anthropic connection failed ({exc.__class__.__name__})", provider="anthropic"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("Failed to parse Anthropic response: %s", exc.__class__.__name__)
            raise ProviderResponseError(
                f"Invalid response from Anthropic: {exc}", provider="anthropic"
            ) from exc

        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        finish_reason = data.get("stop_reason", "end_turn")
        model_used = data.get("model", target_model)

        usage: TokenUsage | None = None
        raw_usage = data.get("usage")
        if raw_usage and isinstance(raw_usage, dict):
            inp = raw_usage.get("input_tokens", 0)
            out = raw_usage.get("output_tokens", 0)
            usage = TokenUsage(
                prompt_tokens=inp,
                completion_tokens=out,
                total_tokens=inp + out,
            )

        logger.info("Anthropic completion succeeded (model=%s)", model_used)
        return ChatResponse(
            content=content,
            model=model_used,
            finish_reason=finish_reason,
            usage=usage,
            provider="anthropic",
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks asynchronously from Anthropic via SSE."""
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/messages"

        system_prompt, messages = self._split_messages(request.messages)
        max_tokens = request.max_tokens or 4096

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        logger.info("Opening Anthropic streaming stream (model=%s)", target_model)

        try:
            async with self._client.stream(
                "POST", endpoint, json=payload, headers=self._headers
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    code = response.status_code
                    text = error_body.decode(errors="replace")
                    logger.error("Anthropic streaming returned HTTP %d", code)
                    if code in (401, 403):
                        raise ProviderResponseError(
                            f"Anthropic authentication error ({code}): {text}",
                            provider="anthropic",
                            status_code=code,
                        )
                    if code == 429:
                        raise ProviderResponseError(
                            f"Anthropic rate limit exceeded ({code}): {text}",
                            provider="anthropic",
                            status_code=code,
                        )
                    raise ProviderResponseError(
                        f"Anthropic streaming error ({code}): {text}",
                        provider="anthropic",
                        status_code=code,
                    )

                # Anthropic documents the flow as: message_start, content blocks, ONE OR MORE
                # message_delta events, then a FINAL message_stop. A stop_reason on a
                # message_delta is therefore not terminal (more deltas or an `error` event may
                # still follow); only message_stop proves the response completed.
                completed = False

                async for line in response.aiter_lines():
                    cleaned_line = line.strip()
                    if not cleaned_line or not cleaned_line.startswith("data:"):
                        continue

                    data_str = cleaned_line[len("data:") :].strip()
                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed Anthropic stream frame: %s", exc)
                        continue
                    if not isinstance(chunk_json, dict):
                        continue

                    event_type = chunk_json.get("type")
                    if event_type == "error":
                        logger.error("Anthropic stream reported an error event")
                        raise _error_event_to_exception(chunk_json)
                    if event_type == "content_block_delta":
                        delta_obj = chunk_json.get("delta", {})
                        if delta_obj.get("type") == "text_delta":
                            yield ChatStreamChunk(
                                delta=delta_obj.get("text", ""),
                                model=target_model,
                                provider="anthropic",
                            )
                    elif event_type == "message_delta":
                        delta_obj = chunk_json.get("delta", {})
                        stop_reason = delta_obj.get("stop_reason")
                        raw_usage = chunk_json.get("usage")
                        usage: TokenUsage | None = None
                        if raw_usage:
                            out = raw_usage.get("output_tokens", 0)
                            usage = TokenUsage(completion_tokens=out, total_tokens=out)
                        yield ChatStreamChunk(
                            delta="",
                            model=target_model,
                            finish_reason=stop_reason,
                            usage=usage,
                            provider="anthropic",
                        )
                    elif event_type == "message_stop":
                        completed = True
                        break

                if not completed:
                    # The connection closed without the final message_stop event: the reply
                    # may be incomplete and must not be treated as a finished generation.
                    logger.error("Anthropic stream ended before message_stop")
                    raise ProviderResponseError(
                        "Anthropic stream ended before message_stop (truncated response)",
                        provider="anthropic",
                    )
        except httpx.TimeoutException as exc:
            logger.error("Anthropic stream timed out")
            raise ProviderTimeoutError(
                f"Anthropic stream timed out after {self.timeout}s", provider="anthropic"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Anthropic stream network connection failed")
            raise ProviderConnectionError(
                "Anthropic stream connection failed", provider="anthropic"
            ) from exc
        except httpx.UnsupportedProtocol as exc:
            raise ProviderConfigurationError(
                f"Invalid Anthropic base URL: {safe_url(self.base_url)}", provider="anthropic"
            ) from exc
        except httpx.TransportError as exc:
            # RemoteProtocolError (dropped / incomplete chunked read), ProxyError, ...
            logger.error("Anthropic stream transport failure: %s", exc.__class__.__name__)
            raise ProviderConnectionError(
                f"Anthropic stream connection dropped ({exc.__class__.__name__})",
                provider="anthropic",
            ) from exc
        except httpx.HTTPError as exc:
            # DecodingError, TooManyRedirects, ...: unusable response from the provider.
            logger.error("Anthropic stream protocol failure: %s", exc.__class__.__name__)
            raise ProviderResponseError(
                f"Invalid Anthropic stream response ({exc.__class__.__name__})",
                provider="anthropic",
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Anthropic."""
        endpoint = f"{self.base_url}/models"
        try:
            response = await self._client.get(endpoint, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Anthropic models list timed out: {exc}", provider="anthropic"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Anthropic at {safe_url(self.base_url)}: {exc}",
                provider="anthropic",
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Failed to retrieve Anthropic models: {exc}",
                provider="anthropic",
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
            ) from exc

        model_items = data.get("data", [])
        return [
            ModelInfo(
                name=m.get("id", "unknown"),
                modified_at=str(m.get("created_at")) if m.get("created_at") else None,
            )
            for m in model_items
            if isinstance(m, dict) and "id" in m
        ]

    async def health_check(self) -> ProviderHealth:
        """Verify Anthropic connectivity and key validity."""
        try:
            models = await self.list_models()
            return ProviderHealth(
                healthy=True,
                provider="anthropic",
                details={
                    "base_url": safe_url(self.base_url),
                    "default_model": self.default_model,
                    "model_count": len(models),
                },
            )
        except ProviderError as exc:
            logger.warning("Anthropic health check failed: %s", describe_error(exc))
            return ProviderHealth(
                healthy=False,
                provider="anthropic",
                details={"base_url": safe_url(self.base_url), "default_model": self.default_model},
                error=describe_error(exc),
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._own_client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed Anthropic HTTP client")
