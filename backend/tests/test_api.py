"""Integration and unit tests for FastAPI endpoints (backend/app/main.py)."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_orchestrator, get_provider
from app.brain.orchestrator import BrainOrchestrator
from app.main import app, lifespan
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConnectionError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    TokenUsage,
)


class MockAPIProvider(LLMProvider):
    """Test double for LLMProvider inside API tests."""

    def __init__(self) -> None:
        self.healthy = True
        self.models = [ModelInfo(name="qwen3.5:4b", size=3400000000)]
        self.simulate_error: Exception | None = None
        self.closed = False

    async def generate(self, request: ChatRequest) -> ChatResponse:
        if self.simulate_error:
            raise self.simulate_error
        return ChatResponse(
            content="Hello from AARYX API!",
            model=request.model or "qwen3.5:4b",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        if self.simulate_error:
            raise self.simulate_error
        yield ChatStreamChunk(delta="Hello ", model="qwen3.5:4b")
        yield ChatStreamChunk(
            delta="world!",
            model="qwen3.5:4b",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    async def list_models(self) -> list[ModelInfo]:
        if self.simulate_error:
            raise self.simulate_error
        return self.models

    async def health_check(self) -> ProviderHealth:
        if not self.healthy:
            return ProviderHealth(healthy=False, provider="mock", error="Mock daemon down")
        return ProviderHealth(
            healthy=True,
            provider="mock",
            details={"model_count": len(self.models), "available_models": ["qwen3.5:4b"]},
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def mock_provider() -> MockAPIProvider:
    return MockAPIProvider()


@pytest.fixture
def client(mock_provider: MockAPIProvider):
    """AsyncClient fixture with dependency overrides for isolated testing."""
    mock_orchestrator = BrainOrchestrator(provider=mock_provider)

    app.dependency_overrides[get_provider] = lambda: mock_provider
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator

    # Also assign to state for any handler that directly queries state
    app.state.provider = mock_provider
    app.state.orchestrator = mock_orchestrator

    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost")

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint_healthy(client: AsyncClient, mock_provider: MockAPIProvider):
    """Verify GET /health returns 200 with operational status."""
    async with client as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "AARYX"
    assert data["provider"]["healthy"] is True
    assert data["provider"]["provider"] == "mock"


@pytest.mark.asyncio
async def test_health_endpoint_degraded(client: AsyncClient, mock_provider: MockAPIProvider):
    """Verify GET /health reports degraded when provider health check fails."""
    mock_provider.healthy = False
    async with client as c:
        response = await c.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["provider"]["healthy"] is False


@pytest.mark.asyncio
async def test_chat_endpoint_success(client: AsyncClient):
    """Verify POST /api/chat produces a validated conversational reply."""
    payload = {
        "message": "Hello AARYX!",
        "history": [{"role": "user", "content": "Prior message"}],
        "temperature": 0.5,
    }
    async with client as c:
        response = await c.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "Hello from AARYX API!"
    assert data["model"] == "qwen3.5:4b"
    assert data["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_endpoint_validation_empty_message(client: AsyncClient):
    """Verify POST /api/chat rejects empty or whitespace-only messages."""
    async with client as c:
        response = await c.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_chat_endpoint_connection_error(client: AsyncClient, mock_provider: MockAPIProvider):
    """Verify ProviderConnectionError maps to HTTP 503."""
    mock_provider.simulate_error = ProviderConnectionError("Daemon unreachable", provider="ollama")
    async with client as c:
        response = await c.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 503
    data = response.json()
    assert data["error"] == "Provider connection failed"
    assert data["provider"] == "ollama"


@pytest.mark.asyncio
async def test_chat_endpoint_timeout_error(client: AsyncClient, mock_provider: MockAPIProvider):
    """Verify ProviderTimeoutError maps to HTTP 504."""
    mock_provider.simulate_error = ProviderTimeoutError("Inference timed out", provider="ollama")
    async with client as c:
        response = await c.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 504
    data = response.json()
    assert data["error"] == "Provider timeout"


@pytest.mark.asyncio
async def test_chat_endpoint_response_error(client: AsyncClient, mock_provider: MockAPIProvider):
    """Verify ProviderResponseError maps to HTTP 502."""
    mock_provider.simulate_error = ProviderResponseError(
        "500 Internal error from Ollama", provider="ollama"
    )
    async with client as c:
        response = await c.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 502
    data = response.json()
    assert data["error"] == "Provider response error"


@pytest.mark.asyncio
async def test_chat_stream_endpoint(client: AsyncClient):
    """Verify POST /api/chat/stream returns real-time SSE chunks."""
    payload = {"message": "Stream this test"}
    async with client as c:
        response = await c.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data: " in body
    assert "Hello " in body
    assert "world!" in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_models_endpoint(client: AsyncClient):
    """Verify GET /api/models returns provider models."""
    async with client as c:
        response = await c.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "qwen3.5:4b"


@pytest.mark.asyncio
async def test_lifespan_lifecycle(tmp_path, monkeypatch):
    """Verify application startup and shutdown lifecycle cleans up provider and memory."""
    from app.core.config import get_settings

    test_db = tmp_path / "lifespan_test.db"
    monkeypatch.setenv("MEMORY_DATABASE_URL", f"sqlite+aiosqlite:///{test_db}")
    get_settings.cache_clear()

    test_application = app
    async with lifespan(test_application):
        assert hasattr(test_application.state, "provider")
        assert hasattr(test_application.state, "orchestrator")
        assert hasattr(test_application.state, "memory_engine")
        assert hasattr(test_application.state, "memory_service")
        assert test_application.state.provider is not None
        assert test_application.state.memory_engine is not None
        assert test_application.state.memory_service is not None
        assert not test_application.state.provider._client.is_closed

    # Verify provider client is closed after shutdown
    assert test_application.state.provider._client.is_closed
    get_settings.cache_clear()
