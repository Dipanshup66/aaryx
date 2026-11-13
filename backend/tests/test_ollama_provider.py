"""Unit tests for OllamaProvider using httpx.MockTransport."""

import json

import httpx
import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.providers.ollama import OllamaProvider


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello AARYX!"),
        ],
        model="qwen3.5:4b",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.mark.asyncio
async def test_generate_success(sample_chat_request):
    """Verify successful non-streaming chat generation and payload formatting."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        captured_request["method"] = request.method
        captured_request["json"] = json.loads(request.content.decode("utf-8"))

        return httpx.Response(
            status_code=200,
            json={
                "model": "qwen3.5:4b",
                "message": {"role": "assistant", "content": "Hello! I am AARYX."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 8,
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    response = await provider.generate(sample_chat_request)

    assert response.content == "Hello! I am AARYX."
    assert response.model == "qwen3.5:4b"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 8
    assert response.usage.total_tokens == 20

    # Verify request payload sent to Ollama
    payload = captured_request["json"]
    assert payload["model"] == "qwen3.5:4b"
    assert payload["stream"] is False
    assert len(payload["messages"]) == 2
    assert payload["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert payload["messages"][1] == {"role": "user", "content": "Hello AARYX!"}
    assert payload["options"]["temperature"] == 0.7
    assert payload["options"]["num_predict"] == 100

    await provider.aclose()


@pytest.mark.asyncio
async def test_stream_success(sample_chat_request):
    """Verify streaming response chunks and usage metrics on completion."""
    chunks = [
        json.dumps({"model": "qwen3.5:4b", "message": {"content": "Hello"}, "done": False}) + "\n",
        json.dumps({"model": "qwen3.5:4b", "message": {"content": " world"}, "done": False}) + "\n",
        json.dumps(
            {
                "model": "qwen3.5:4b",
                "message": {"content": "!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 3,
            }
        )
        + "\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["stream"] is True

        return httpx.Response(
            status_code=200,
            content="".join(chunks).encode("utf-8"),
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    collected_deltas = []
    final_chunk = None

    async for chunk in provider.stream(sample_chat_request):
        collected_deltas.append(chunk.delta)
        if chunk.finish_reason is not None:
            final_chunk = chunk

    assert "".join(collected_deltas) == "Hello world!"
    assert final_chunk is not None
    assert final_chunk.finish_reason == "stop"
    assert final_chunk.usage is not None
    assert final_chunk.usage.prompt_tokens == 10
    assert final_chunk.usage.completion_tokens == 3
    assert final_chunk.usage.total_tokens == 13

    await provider.aclose()


@pytest.mark.asyncio
async def test_list_models_success():
    """Verify list_models maps Ollama /api/tags JSON to ModelInfo list."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            status_code=200,
            json={
                "models": [
                    {
                        "name": "qwen3.5:4b",
                        "size": 3389983735,
                        "modified_at": "2026-06-11T03:10:49Z",
                        "details": {"format": "gguf", "parameter_size": "4.7B"},
                    },
                    {
                        "name": "nomic-embed-text:latest",
                        "size": 274302450,
                        "modified_at": "2026-06-14T11:02:13Z",
                        "details": {"format": "gguf", "parameter_size": "137M"},
                    },
                ]
            },
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    models = await provider.list_models()
    assert len(models) == 2
    assert models[0].name == "qwen3.5:4b"
    assert models[0].size == 3389983735
    assert models[0].details["parameter_size"] == "4.7B"
    assert models[1].name == "nomic-embed-text:latest"

    await provider.aclose()


@pytest.mark.asyncio
async def test_health_check_healthy():
    """Verify health_check reports healthy=True when Ollama responds 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"models": [{"name": "qwen3.5:4b"}]})

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    health = await provider.health_check()
    assert health.healthy is True
    assert health.provider == "ollama"
    assert health.details["model_count"] == 1
    assert "qwen3.5:4b" in health.details["available_models"]

    await provider.aclose()


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Verify health_check reports healthy=False without throwing unhandled exceptions."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by daemon")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    health = await provider.health_check()
    assert health.healthy is False
    assert health.provider == "ollama"
    assert "Connection refused" in health.error

    await provider.aclose()


@pytest.mark.asyncio
async def test_connection_error_raises_provider_connection_error(sample_chat_request):
    """Verify connection error triggers ProviderConnectionError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Network unreachable")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://unreachable:11434", client=mock_client)

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.generate(sample_chat_request)
    assert "Cannot connect to Ollama daemon" in str(exc_info.value)
    assert exc_info.value.provider == "ollama"

    await provider.aclose()


@pytest.mark.asyncio
async def test_timeout_error_raises_provider_timeout_error(sample_chat_request):
    """Verify timeout triggers ProviderTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Generation timed out")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(sample_chat_request)
    assert "timed out" in str(exc_info.value)
    assert exc_info.value.provider == "ollama"

    await provider.aclose()


@pytest.mark.asyncio
async def test_invalid_json_raises_provider_response_error(sample_chat_request):
    """Verify malformed JSON responses trigger ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, content=b"Non-JSON response from proxy")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)
    assert "Invalid response" in str(exc_info.value)

    await provider.aclose()


@pytest.mark.asyncio
async def test_http_error_status_raises_provider_response_error(sample_chat_request):
    """Verify 4xx/5xx status codes trigger ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=404, text="model 'unknown-model' not found")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)
    assert "HTTP 404" in str(exc_info.value)

    await provider.aclose()


@pytest.mark.asyncio
async def test_streaming_error_status(sample_chat_request):
    """Verify streaming handles non-200 responses with ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, text="Internal Ollama failure")

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider(base_url="http://mock-ollama:11434", client=mock_client)

    with pytest.raises(ProviderResponseError) as exc_info:
        async for _ in provider.stream(sample_chat_request):
            pass
    assert "Ollama streaming returned HTTP 500" in str(exc_info.value)

    await provider.aclose()


@pytest.mark.asyncio
async def test_context_manager_lifecycle():
    """Verify async context manager properly creates and cleans up internal client."""
    async with OllamaProvider(base_url="http://mock-ollama:11434") as provider:
        assert provider._own_client is True
        assert not provider._client.is_closed

    assert provider._client.is_closed
