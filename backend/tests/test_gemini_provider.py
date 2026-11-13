"""Unit tests for GeminiProvider using httpx.MockTransport."""

import json

import httpx
import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.providers.gemini import GeminiProvider


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are a personal assistant."),
            ChatMessage(role="user", content="Hello Gemini!"),
        ],
        model="gemini-1.5-flash",
        temperature=0.4,
        max_tokens=300,
    )


def test_missing_api_key_raises_error():
    """Verify initialization fails when Gemini API key is missing."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        GeminiProvider(api_key="", client=httpx.AsyncClient())
    assert "Gemini API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_generate_success(sample_chat_request):
    """Verify successful non-streaming chat generation and payload mapping."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-1.5-flash:generateContent"
        assert request.headers["x-goog-api-key"] == "test-gemini-key"
        captured_request["json"] = json.loads(request.content.decode("utf-8"))

        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello from Google Gemini!"}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 16,
                    "candidatesTokenCount": 8,
                    "totalTokenCount": 24,
                },
                "modelVersion": "gemini-1.5-flash",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    response = await provider.generate(sample_chat_request)

    assert response.content == "Hello from Google Gemini!"
    assert response.model == "gemini-1.5-flash"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 16
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 24

    payload = captured_request["json"]
    assert payload["systemInstruction"]["parts"][0]["text"] == "You are a personal assistant."
    assert payload["generationConfig"]["temperature"] == 0.4
    assert payload["generationConfig"]["maxOutputTokens"] == 300
    assert len(payload["contents"]) == 1
    assert payload["contents"][0]["role"] == "user"
    assert payload["contents"][0]["parts"][0]["text"] == "Hello Gemini!"


@pytest.mark.asyncio
async def test_stream_success(sample_chat_request):
    """Verify streaming response chunks from Gemini SSE stream."""
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}\n\n'
        b'data: {"candidates":[{"content":{"parts":[{"text":"Gemini!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/models/gemini-1.5-flash:streamGenerateContent" in request.url.path
        assert request.url.query.decode("utf-8") == "alt=sse"
        return httpx.Response(status_code=200, content=sse_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    chunks = []
    async for chunk in provider.stream(sample_chat_request):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "Hello "
    assert chunks[1].delta == "Gemini!"
    assert chunks[1].finish_reason == "stop"
    assert chunks[1].usage is not None
    assert chunks[1].usage.prompt_tokens == 10
    assert chunks[1].usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_authentication_error(sample_chat_request):
    """Verify 400/403 API key error raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            json={
                "error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="invalid-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "authentication or request error (400)" in str(exc_info.value).lower()
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_rate_limit_error(sample_chat_request):
    """Verify 429 Resource Exhausted raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={
                "error": {
                    "code": 429,
                    "message": "Resource has been exhausted",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "rate limit or quota exceeded" in str(exc_info.value).lower()
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_timeout_error(sample_chat_request):
    """Verify timeout during request raises ProviderTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "timed out" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_connection_error(sample_chat_request):
    """Verify network connection failure raises ProviderConnectionError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "Cannot connect" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_malformed_response(sample_chat_request):
    """Verify missing candidates raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"promptFeedback": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "contained no candidates" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_list_models_success():
    """Verify models listing endpoint strips 'models/' prefix and maps to ModelInfo."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models"
        return httpx.Response(
            status_code=200,
            json={
                "models": [
                    {
                        "name": "models/gemini-1.5-flash",
                        "displayName": "Gemini 1.5 Flash",
                        "description": "Fast model",
                    },
                    {
                        "name": "models/gemini-1.5-pro",
                        "displayName": "Gemini 1.5 Pro",
                        "description": "Complex reasoning model",
                    },
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    models = await provider.list_models()
    assert len(models) == 2
    names = [m.name for m in models]
    assert "gemini-1.5-flash" in names
    assert "gemini-1.5-pro" in names


@pytest.mark.asyncio
async def test_health_check_healthy():
    """Verify health check returns healthy when models query succeeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"models": [{"name": "models/gemini-1.5-flash"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    health = await provider.health_check()
    assert health.healthy is True
    assert health.provider == "gemini"
    assert health.details["model_count"] == 1


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Verify health check returns unhealthy when models query fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=400, json={"error": "Invalid API key"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GeminiProvider(api_key="test-gemini-key", client=client)

    health = await provider.health_check()
    assert health.healthy is False
    assert health.provider == "gemini"
    assert health.error is not None
