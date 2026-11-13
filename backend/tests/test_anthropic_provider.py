"""Unit tests for AnthropicProvider using httpx.MockTransport."""

import json

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello Claude!"),
        ],
        model="claude-3-5-sonnet-20241022",
        temperature=0.5,
        max_tokens=200,
    )


def test_missing_api_key_raises_error():
    """Verify initialization fails when Anthropic API key is missing."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        AnthropicProvider(api_key="", client=httpx.AsyncClient())
    assert "Anthropic API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_generate_success(sample_chat_request):
    """Verify successful non-streaming chat generation and payload formatting."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "test-anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        captured_request["json"] = json.loads(request.content.decode("utf-8"))

        return httpx.Response(
            status_code=200,
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Anthropic Claude!"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 18,
                    "output_tokens": 7,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    response = await provider.generate(sample_chat_request)

    assert response.content == "Hello from Anthropic Claude!"
    assert response.model == "claude-3-5-sonnet-20241022"
    assert response.finish_reason == "end_turn"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 18
    assert response.usage.completion_tokens == 7
    assert response.usage.total_tokens == 25

    # System message must NOT be in messages array, but in top-level system parameter
    payload = captured_request["json"]
    assert payload["system"] == "You are a helpful assistant."
    assert payload["max_tokens"] == 200
    assert payload["temperature"] == 0.5
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello Claude!"


@pytest.mark.asyncio
async def test_stream_success(sample_chat_request):
    """Verify streaming response chunks from Anthropic SSE stream."""
    sse_body = (
        b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}\n\n'
        b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Claude!"}}\n\n'
        b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}\n\n'
        b'data: {"type": "message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        return httpx.Response(status_code=200, content=sse_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    chunks = []
    async for chunk in provider.stream(sample_chat_request):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].delta == "Hello "
    assert chunks[1].delta == "Claude!"
    assert chunks[2].finish_reason == "end_turn"
    assert chunks[2].usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_authentication_error(sample_chat_request):
    """Verify 401 Unauthorized raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"error": {"type": "authentication_error", "message": "Invalid API Key"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="invalid-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "authentication error (401)" in str(exc_info.value).lower()
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_rate_limit_error(sample_chat_request):
    """Verify 429 Too Many Requests raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "rate limit" in str(exc_info.value).lower()
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_timeout_error(sample_chat_request):
    """Verify timeout during request raises ProviderTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "timed out" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_connection_error(sample_chat_request):
    """Verify network connection failure raises ProviderConnectionError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "Cannot connect" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_malformed_response(sample_chat_request):
    """Verify non-JSON payload raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="Service Unavailable Gateway")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "Invalid response" in str(exc_info.value)
    assert exc_info.value.provider == "anthropic"


@pytest.mark.asyncio
async def test_list_models_success():
    """Verify models listing endpoint maps Anthropic format into ModelInfo domain models."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"id": "claude-3-5-sonnet-20241022", "created_at": "2024-10-22"},
                    {"id": "claude-3-haiku-20240307", "created_at": "2024-03-07"},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    models = await provider.list_models()
    assert len(models) == 2
    names = [m.name for m in models]
    assert "claude-3-5-sonnet-20241022" in names
    assert "claude-3-haiku-20240307" in names


@pytest.mark.asyncio
async def test_health_check_healthy():
    """Verify health check returns healthy when models query succeeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={"data": [{"id": "claude-3-5-sonnet-20241022"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    health = await provider.health_check()
    assert health.healthy is True
    assert health.provider == "anthropic"
    assert health.details["model_count"] == 1


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Verify health check returns unhealthy when models query fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, json={"error": "Unauthorized"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = AnthropicProvider(api_key="test-anthropic-key", client=client)

    health = await provider.health_check()
    assert health.healthy is False
    assert health.provider == "anthropic"
    assert health.error is not None
