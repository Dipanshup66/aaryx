"""Ollama local AI provider adapter.

Communicates with the local Ollama instance via its native HTTP API:
- POST /api/chat (non-streaming and streaming completions)
- GET /api/tags (model listing and health check)
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConnectionError,
    ProviderError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    TokenUsage,
    redact_secrets,
    safe_url,
)

logger = get_logger("providers.ollama")


class OllamaProvider(LLMProvider):
    """Local LLM provider connecting to Ollama daemon."""

    def __init__(
        self,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 180.0,
        connect_timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Base URL for Ollama daemon. Defaults to Settings.ollama_base_url.
            default_model: Target model name. Defaults to Settings.ollama_model.
            timeout: Total read/write timeout in seconds for inference.
            connect_timeout: Connection establishment timeout in seconds.
            client: Optional pre-configured httpx.AsyncClient (e.g. for testing).
        """
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).strip().rstrip("/")
        self.default_model = default_model or settings.ollama_model
        self.timeout = timeout
        self.connect_timeout = connect_timeout

        self._own_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=self.timeout,
                connect=self.connect_timeout,
            )
        )

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict[str, str]]:
        """Format domain messages into Ollama's expected JSON array."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _build_options(self, request: ChatRequest) -> dict[str, Any]:
        """Map optional parameters into Ollama options payload."""
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        return options

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete chat completion from Ollama.

        Args:
            request: The conversational request.

        Returns:
            The complete ChatResponse.

        Raises:
            ProviderConnectionError: If Ollama cannot be reached.
            ProviderTimeoutError: If the request times out.
            ProviderResponseError: If Ollama returns an error code or invalid format.
        """
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._format_messages(request.messages),
            "stream": False,
        }
        options = self._build_options(request)
        if options:
            payload["options"] = options

        logger.info("Dispatching non-streaming completion request (model=%s)", target_model)

        try:
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out after %.1fs", self.timeout)
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self.timeout}s", provider="ollama"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to Ollama at %s", safe_url(self.base_url))
            raise ProviderConnectionError(
                f"Cannot connect to Ollama daemon at {safe_url(self.base_url)}", provider="ollama"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama returned HTTP error status %d", exc.response.status_code)
            raise ProviderResponseError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}",
                provider="ollama",
                status_code=exc.response.status_code,
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse Ollama response: %s", exc.__class__.__name__)
            raise ProviderResponseError(
                f"Invalid response from Ollama: {exc}", provider="ollama"
            ) from exc

        # Extract message content and metadata
        message_obj = data.get("message", {})
        content = message_obj.get("content", "")
        thinking = message_obj.get("thinking")
        model_used = data.get("model", target_model)
        finish_reason = data.get("done_reason", "stop")

        usage: TokenUsage | None = None
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            p_tok = prompt_tokens or 0
            c_tok = completion_tokens or 0
            usage = TokenUsage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
            )

        logger.info(
            "Completion succeeded (model=%s, prompt_tokens=%s, completion_tokens=%s)",
            model_used,
            prompt_tokens,
            completion_tokens,
        )

        return ChatResponse(
            content=content,
            model=model_used,
            finish_reason=finish_reason,
            usage=usage,
            thinking=thinking,
            provider="ollama",
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks asynchronously from Ollama.

        Yields:
            ChatStreamChunk containing incremental token deltas.

        Raises:
            ProviderConnectionError: If connection fails.
            ProviderTimeoutError: If connection or chunk read times out.
            ProviderResponseError: If stream fails or response is not 200.
        """
        target_model = request.model or self.default_model
        endpoint = f"{self.base_url}/api/chat"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": self._format_messages(request.messages),
            "stream": True,
        }
        options = self._build_options(request)
        if options:
            payload["options"] = options

        logger.info("Opening streaming completion stream (model=%s)", target_model)

        try:
            async with self._client.stream("POST", endpoint, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise ProviderResponseError(
                        f"Ollama streaming returned HTTP {response.status_code}: {error_body.decode(errors='replace')}",
                        provider="ollama",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue

                    try:
                        chunk_json = json.loads(cleaned_line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed streaming JSON chunk: %s", exc)
                        continue

                    delta_text = chunk_json.get("message", {}).get("content", "")
                    thinking_text = chunk_json.get("message", {}).get("thinking")
                    done = chunk_json.get("done", False)

                    # Skip empty heartbeats if no content, thinking, or termination signal
                    if not delta_text and not thinking_text and not done:
                        continue

                    usage: TokenUsage | None = None
                    finish_reason: str | None = None

                    if done:
                        finish_reason = chunk_json.get("done_reason", "stop")
                        p_tok = chunk_json.get("prompt_eval_count") or 0
                        c_tok = chunk_json.get("eval_count") or 0
                        if p_tok or c_tok:
                            usage = TokenUsage(
                                prompt_tokens=p_tok,
                                completion_tokens=c_tok,
                                total_tokens=p_tok + c_tok,
                            )

                    yield ChatStreamChunk(
                        delta=delta_text,
                        model=chunk_json.get("model", target_model),
                        finish_reason=finish_reason,
                        usage=usage,
                        thinking=thinking_text,
                        provider="ollama",
                    )

        except httpx.TimeoutException as exc:
            logger.error("Streaming from Ollama timed out (%s)", exc.__class__.__name__)
            raise ProviderTimeoutError(
                f"Ollama streaming timed out: {exc}", provider="ollama"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error(
                "Connection lost during streaming from Ollama (%s)", exc.__class__.__name__
            )
            raise ProviderConnectionError(
                f"Connection error with Ollama daemon: {exc}", provider="ollama"
            ) from exc
        except ProviderError:
            raise
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as exc:
            logger.error("Unexpected error during Ollama streaming (%s)", exc.__class__.__name__)
            raise ProviderResponseError(
                f"Unexpected error during streaming: {exc}", provider="ollama"
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        """Retrieve the list of models currently available in local Ollama."""
        endpoint = f"{self.base_url}/api/tags"
        try:
            response = await self._client.get(endpoint)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                "Timed out listing Ollama models", provider="ollama"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderConnectionError(
                f"Failed to connect to Ollama at {safe_url(self.base_url)}", provider="ollama"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Failed to retrieve model list: {exc}",
                provider="ollama",
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
            ) from exc

        models_raw = data.get("models", [])
        return [
            ModelInfo(
                name=m.get("name", "unknown"),
                size=m.get("size"),
                modified_at=m.get("modified_at"),
                details=m.get("details"),
            )
            for m in models_raw
        ]

    async def health_check(self) -> ProviderHealth:
        """Perform a non-intrusive health check against Ollama daemon."""
        endpoint = f"{self.base_url}/api/tags"
        try:
            # Use short timeout for health ping
            response = await self._client.get(endpoint, timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return ProviderHealth(
                    healthy=True,
                    provider="ollama",
                    details={
                        "base_url": safe_url(self.base_url),
                        "default_model": self.default_model,
                        "model_count": len(models),
                        "available_models": models,
                    },
                )
            return ProviderHealth(
                healthy=False,
                provider="ollama",
                error=f"HTTP {response.status_code}",
            )
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as exc:
            return ProviderHealth(
                healthy=False,
                provider="ollama",
                error=(
                    f"Unreachable at {safe_url(self.base_url)} "
                    f"({exc.__class__.__name__}): {redact_secrets(str(exc))[:200]}"
                ),
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if managed by this provider."""
        if self._own_client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed Ollama HTTP client")
