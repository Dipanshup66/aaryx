"""Google Gemini cloud AI provider adapter.

Communicates with Google Gemini REST API endpoints:
- POST /v1beta/models/{model}:generateContent
- POST /v1beta/models/{model}:streamGenerateContent?alt=sse
- GET /v1beta/models
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import SecretStr

from app.core.config import get_settings
from app.core.logging import get_logger
from app.providers.base import (
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

logger = get_logger("providers.gemini")


class GeminiProvider(LLMProvider):
    """Cloud LLM provider adapter connecting to Google Gemini REST API."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        connect_timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Gemini provider with credential validation and HTTP client setup."""
        settings = get_settings()

        resolved_key = api_key if api_key is not None else settings.gemini_api_key
        if isinstance(resolved_key, SecretStr):
            raw_key = resolved_key.get_secret_value()
        elif isinstance(resolved_key, str):
            raw_key = resolved_key.strip()
        else:
            raw_key = ""

        if not raw_key:
            raise ProviderConfigurationError(
                "Gemini API key is missing or empty. Please configure GEMINI_API_KEY.",
                provider="gemini",
            )

        self._api_key = raw_key
        self.base_url = (base_url or settings.gemini_base_url).strip().rstrip("/")
        self.default_model = default_model or settings.gemini_model
        self.timeout = timeout if timeout is not None else settings.gemini_timeout
        self.connect_timeout = connect_timeout

        self._headers = {
            "x-goog-api-key": self._api_key,
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

    def _format_payload(
        self, request: ChatRequest
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
        """Format domain messages into Gemini's systemInstruction, contents, and generationConfig."""
        system_prompts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == "system":
                system_prompts.append(msg.content)
            else:
                # Gemini maps assistant role to 'model'
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append(
                    {
                        "role": gemini_role,
                        "parts": [{"text": msg.content}],
                    }
                )

        system_instruction = (
            {"parts": [{"text": "\n\n".join(system_prompts)}]} if system_prompts else None
        )

        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens

        return system_instruction, contents, generation_config

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete non-streaming chat completion from Google Gemini."""
        target_model = request.model or self.default_model
        # Strip any accidental 'models/' prefix to avoid double prefixes
        clean_model = target_model.removeprefix("models/")
        endpoint = f"{self.base_url}/models/{clean_model}:generateContent"

        system_instruction, contents, generation_config = self._format_payload(request)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if generation_config:
            payload["generationConfig"] = generation_config

        logger.info("Dispatching Gemini completion request (model=%s)", clean_model)

        try:
            response = await self._client.post(endpoint, json=payload, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            logger.error("Gemini request timed out after %.1fs", self.timeout)
            raise ProviderTimeoutError(
                f"Gemini request timed out after {self.timeout}s", provider="gemini"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Failed to connect to Gemini at %s", safe_url(self.base_url))
            raise ProviderConnectionError(
                f"Cannot connect to Gemini service at {safe_url(self.base_url)}", provider="gemini"
            ) from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            text = exc.response.text
            logger.error("Gemini returned HTTP error status %d", code)
            if code in (400, 401, 403):
                raise ProviderResponseError(
                    f"Gemini authentication or request error ({code}): {text}",
                    provider="gemini",
                    status_code=code,
                ) from exc
            if code == 429:
                raise ProviderResponseError(
                    f"Gemini rate limit or quota exceeded ({code}): {text}",
                    provider="gemini",
                    status_code=code,
                ) from exc
            raise ProviderResponseError(
                f"Gemini returned HTTP {code}: {text}", provider="gemini", status_code=code
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("Failed to parse Gemini response: %s", exc.__class__.__name__)
            raise ProviderResponseError(
                f"Invalid response from Gemini: {exc}", provider="gemini"
            ) from exc

        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderResponseError(
                "Gemini response contained no candidates", provider="gemini"
            )

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        content = "".join(part.get("text", "") for part in parts)
        finish_reason = candidate.get("finishReason", "STOP").lower()
        model_used = data.get("modelVersion", clean_model)

        usage: TokenUsage | None = None
        raw_usage = data.get("usageMetadata")
        if raw_usage and isinstance(raw_usage, dict):
            inp = raw_usage.get("promptTokenCount", 0)
            out = raw_usage.get("candidatesTokenCount", 0)
            total = raw_usage.get("totalTokenCount", inp + out)
            usage = TokenUsage(
                prompt_tokens=inp,
                completion_tokens=out,
                total_tokens=total,
            )

        logger.info("Gemini completion succeeded (model=%s)", model_used)
        return ChatResponse(
            content=content,
            model=model_used,
            finish_reason=finish_reason,
            usage=usage,
            provider="gemini",
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks asynchronously via SSE (?alt=sse)."""
        target_model = request.model or self.default_model
        clean_model = target_model.removeprefix("models/")
        endpoint = f"{self.base_url}/models/{clean_model}:streamGenerateContent?alt=sse"

        system_instruction, contents, generation_config = self._format_payload(request)

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if generation_config:
            payload["generationConfig"] = generation_config

        logger.info("Opening Gemini streaming stream (model=%s)", clean_model)

        try:
            async with self._client.stream(
                "POST", endpoint, json=payload, headers=self._headers
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    code = response.status_code
                    text = error_body.decode(errors="replace")
                    logger.error("Gemini streaming returned HTTP %d", code)
                    if code in (400, 401, 403):
                        raise ProviderResponseError(
                            f"Gemini authentication error ({code}): {text}",
                            provider="gemini",
                            status_code=code,
                        )
                    if code == 429:
                        raise ProviderResponseError(
                            f"Gemini rate limit exceeded ({code}): {text}",
                            provider="gemini",
                            status_code=code,
                        )
                    raise ProviderResponseError(
                        f"Gemini streaming error ({code}): {text}",
                        provider="gemini",
                        status_code=code,
                    )

                async for line in response.aiter_lines():
                    cleaned_line = line.strip()
                    if not cleaned_line or not cleaned_line.startswith("data:"):
                        continue

                    data_str = cleaned_line[len("data:") :].strip()
                    try:
                        chunk_json = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed Gemini stream frame: %s", exc)
                        continue

                    candidates = chunk_json.get("candidates", [])
                    if not candidates:
                        continue

                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    delta_text = "".join(part.get("text", "") for part in parts)
                    raw_finish = candidate.get("finishReason")
                    finish_reason = raw_finish.lower() if raw_finish else None

                    raw_usage = chunk_json.get("usageMetadata")
                    usage: TokenUsage | None = None
                    if raw_usage:
                        inp = raw_usage.get("promptTokenCount", 0)
                        out = raw_usage.get("candidatesTokenCount", 0)
                        total = raw_usage.get("totalTokenCount", inp + out)
                        usage = TokenUsage(
                            prompt_tokens=inp, completion_tokens=out, total_tokens=total
                        )

                    if not delta_text and finish_reason is None and usage is None:
                        continue

                    yield ChatStreamChunk(
                        delta=delta_text,
                        model=clean_model,
                        finish_reason=finish_reason,
                        usage=usage,
                        provider="gemini",
                    )
        except httpx.TimeoutException as exc:
            logger.error("Gemini stream timed out")
            raise ProviderTimeoutError(
                f"Gemini stream timed out after {self.timeout}s", provider="gemini"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            logger.error("Gemini stream network connection failed")
            raise ProviderConnectionError(
                "Gemini stream connection failed", provider="gemini"
            ) from exc

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from Gemini."""
        endpoint = f"{self.base_url}/models"
        try:
            response = await self._client.get(endpoint, headers=self._headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Gemini models list timed out: {exc}", provider="gemini"
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderConnectionError(
                f"Cannot connect to Gemini at {safe_url(self.base_url)}: {exc}", provider="gemini"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderResponseError(
                f"Failed to retrieve Gemini models: {exc}",
                provider="gemini",
                status_code=getattr(getattr(exc, "response", None), "status_code", None),
            ) from exc

        model_items = data.get("models", [])
        clean_models: list[ModelInfo] = []
        for m in model_items:
            if isinstance(m, dict) and "name" in m:
                raw_name = m["name"]
                model_name = raw_name.removeprefix("models/")
                clean_models.append(
                    ModelInfo(
                        name=model_name,
                        details={
                            "displayName": m.get("displayName"),
                            "description": m.get("description"),
                        },
                    )
                )
        return clean_models

    async def health_check(self) -> ProviderHealth:
        """Verify Gemini connectivity and key validity."""
        try:
            models = await self.list_models()
            return ProviderHealth(
                healthy=True,
                provider="gemini",
                details={
                    "base_url": safe_url(self.base_url),
                    "default_model": self.default_model,
                    "model_count": len(models),
                },
            )
        except ProviderError as exc:
            logger.warning("Gemini health check failed: %s", describe_error(exc))
            return ProviderHealth(
                healthy=False,
                provider="gemini",
                details={"base_url": safe_url(self.base_url), "default_model": self.default_model},
                error=describe_error(exc),
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._own_client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("Closed Gemini HTTP client")
