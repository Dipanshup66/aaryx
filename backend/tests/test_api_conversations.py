"""Integration tests for conversation management and memory-integrated chat endpoints."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.deps import get_memory_service, get_orchestrator, get_provider
from app.brain.orchestrator import BrainOrchestrator
from app.main import app
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.service import MemoryService
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
    ProviderResponseError,
    TokenUsage,
)


class MockMemoryLLMProvider(LLMProvider):
    """Mock LLM Provider for conversation endpoint testing."""

    def __init__(self) -> None:
        self.fail_mid_stream = False
        self.last_received_messages: list[str] = []
        self.thinking_to_inject: str | None = None
        self.stream_thinking_to_inject: list[str] | None = None

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.last_received_messages = [m.content for m in request.messages]
        return ChatResponse(
            content="AI generated memory response",
            model="qwen3.5:4b",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=467, total_tokens=479),
            provider="mock_memory_provider",
            thinking=self.thinking_to_inject,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.last_received_messages = [m.content for m in request.messages]
        if self.stream_thinking_to_inject:
            for think_text in self.stream_thinking_to_inject:
                yield ChatStreamChunk(
                    delta="",
                    model="qwen3.5:4b",
                    thinking=think_text,
                    provider="mock_memory_provider",
                )
        yield ChatStreamChunk(
            delta="Streaming ",
            model="qwen3.5:4b",
            provider="mock_memory_provider",
        )
        if self.fail_mid_stream:
            raise ProviderResponseError("Mid-stream connection dropped", provider="mock")
        yield ChatStreamChunk(
            delta="response content.",
            model="qwen3.5:4b",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16),
            provider="mock_memory_provider",
        )

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="qwen3.5:4b", size=3400000000)]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="mock_memory_provider")

    async def aclose(self) -> None:
        pass


@pytest_asyncio.fixture
async def memory_test_env(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncEngine, MemoryService, MockMemoryLLMProvider]]:
    """Fixture providing isolated temporary database, MemoryService, and Mock LLM."""
    db_file = tmp_path / "api_test_memory.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_database_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)
    memory_service = MemoryService(session_factory=session_factory, max_history_messages=5)
    mock_provider = MockMemoryLLMProvider()

    try:
        yield engine, memory_service, mock_provider
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def api_client(
    memory_test_env: tuple[AsyncEngine, MemoryService, MockMemoryLLMProvider],
) -> AsyncIterator[tuple[AsyncClient, MockMemoryLLMProvider, MemoryService]]:
    """AsyncClient fixture with memory and provider dependency overrides."""
    _, memory_service, mock_provider = memory_test_env
    orchestrator = BrainOrchestrator(provider=mock_provider, max_history_messages=5)

    app.dependency_overrides[get_provider] = lambda: mock_provider
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_memory_service] = lambda: memory_service

    app.state.provider = mock_provider
    app.state.orchestrator = orchestrator
    app.state.memory_service = memory_service

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
    ) as client:
        yield client, mock_provider, memory_service

    app.dependency_overrides.clear()
    app.state.memory_service = None


@pytest.mark.asyncio
async def test_create_and_get_conversation_endpoints(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify POST /api/conversations and GET /api/conversations/{id}."""
    client, _, _ = api_client

    create_payload = {"title": "Architecture Discussions", "metadata": {"topic": "memory"}}
    response = await client.post("/api/conversations", json=create_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["id"] is not None
    assert data["title"] == "Architecture Discussions"
    assert data["metadata_json"] == {"topic": "memory"}

    conv_id = data["id"]

    # Retrieve created conversation
    get_res = await client.get(f"/api/conversations/{conv_id}")
    assert get_res.status_code == 200
    detail = get_res.json()
    assert detail["id"] == conv_id
    assert detail["title"] == "Architecture Discussions"
    assert detail["messages"] == []


@pytest.mark.asyncio
async def test_list_and_delete_conversations_endpoints(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify GET /api/conversations and DELETE /api/conversations/{id}."""
    client, _, _ = api_client

    # Create 2 conversations
    res1 = await client.post("/api/conversations", json={"title": "Thread A"})
    res2 = await client.post("/api/conversations", json={"title": "Thread B"})
    assert res1.status_code == 201
    assert res2.status_code == 201
    id_a = res1.json()["id"]

    # List conversations
    list_res = await client.get("/api/conversations?limit=10&offset=0")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 2
    assert len(list_data["conversations"]) >= 2

    # Delete conversation A
    del_res = await client.delete(f"/api/conversations/{id_a}")
    assert del_res.status_code == 200
    assert del_res.json()["id"] == id_a

    # Verify conversation A no longer exists
    get_deleted = await client.get(f"/api/conversations/{id_a}")
    assert get_deleted.status_code == 404

    # Delete nonexistent returns 404
    del_missing = await client.delete("/api/conversations/nonexistent-id")
    assert del_missing.status_code == 404


@pytest.mark.asyncio
async def test_chat_auto_creates_conversation_and_persists(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify POST /api/chat auto-creates a conversation and stores both user and assistant messages."""
    client, _, memory_service = api_client

    payload = {"message": "Hello AARYX, start a new chat"}
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["reply"] == "AI generated memory response"
    assert data["conversation_id"] is not None

    conv_id = data["conversation_id"]

    # Inspect stored conversation messages via memory service
    conv = await memory_service.get_conversation(conv_id, include_messages=True)
    assert conv is not None
    assert conv.message_count == 2
    messages = await memory_service.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello AARYX, start a new chat"
    assert messages[1].role == "assistant"
    assert messages[1].content == "AI generated memory response"
    assert messages[1].provider == "mock_memory_provider"


@pytest.mark.asyncio
async def test_chat_multi_turn_history_continuation(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify continuing an existing conversation retrieves stored history for context."""
    client, mock_provider, _ = api_client

    # First turn: create conversation
    res1 = await client.post("/api/chat", json={"message": "My name is Alice."})
    assert res1.status_code == 200
    conv_id = res1.json()["conversation_id"]

    # Second turn: use returned conversation_id
    res2 = await client.post(
        "/api/chat",
        json={"conversation_id": conv_id, "message": "What is my name?"},
    )
    assert res2.status_code == 200
    assert res2.json()["conversation_id"] == conv_id

    # Verify that the orchestrator received the previous turns in the messages context
    # Messages sent to LLM should include: System Persona, "My name is Alice.", AI reply, "What is my name?"
    assert "My name is Alice." in mock_provider.last_received_messages
    assert "AI generated memory response" in mock_provider.last_received_messages
    assert "What is my name?" in mock_provider.last_received_messages


@pytest.mark.asyncio
async def test_chat_invalid_conversation_id_returns_404(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify POST /api/chat with an invalid conversation_id returns 404 Not Found."""
    client, _, _ = api_client

    payload = {"conversation_id": "nonexistent-conversation", "message": "Hello?"}
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_chat_stream_persistence(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify POST /api/chat/stream persists user message and completed assistant reply."""
    client, _, memory_service = api_client

    payload = {"message": "Stream and persist this message"}
    response = await client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data: " in body
    assert "Streaming " in body
    assert "response content." in body
    assert "data: [DONE]" in body

    # Extract conversation_id from any stream chunk
    import json

    first_line = next(line for line in body.split("\n") if line.startswith("data: {"))
    chunk_json = json.loads(first_line.removeprefix("data: "))
    conv_id = chunk_json["conversation_id"]
    assert conv_id is not None

    # Verify both user message and completed response are in database
    messages = await memory_service.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Stream and persist this message"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Streaming response content."


@pytest.mark.asyncio
async def test_chat_stream_interrupted_does_not_save_assistant_reply(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify that an interrupted stream saves user prompt but does NOT save partial reply."""
    client, mock_provider, memory_service = api_client
    mock_provider.fail_mid_stream = True

    # Pre-create a conversation
    conv = await memory_service.create_conversation(title="Interrupted Stream Thread")

    payload = {
        "conversation_id": conv.id,
        "message": "This stream will fail midway",
    }
    with pytest.raises((ProviderResponseError, RuntimeError)):
        await client.post("/api/chat/stream", json=payload)

    # Verify database contents: user message exists, but NO assistant message was saved
    messages = await memory_service.get_messages(conv.id)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "This stream will fail midway"


@pytest.mark.asyncio
async def test_memory_disabled_endpoints() -> None:
    """Verify conversation endpoints return 503 when memory service is disabled."""
    app.dependency_overrides[get_memory_service] = lambda: None
    app.state.memory_service = None

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as client:
        res_post = await client.post("/api/conversations", json={"title": "Disabled"})
        assert res_post.status_code == 503
        assert "disabled" in res_post.json()["detail"].lower()

        res_get = await client.get("/api/conversations")
        assert res_get.status_code == 503

        res_detail = await client.get("/api/conversations/some-id")
        assert res_detail.status_code == 503

        res_del = await client.delete("/api/conversations/some-id")
        assert res_del.status_code == 503

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_raw_thinking_not_exposed_in_api_or_database(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify raw thinking is never exposed in POST /api/chat response or stored in database."""
    client, mock_provider, memory_service = api_client
    mock_provider.thinking_to_inject = (
        "Internal prompt analysis: User is testing assistant reasoning. Do not expose this!"
    )

    response = await client.post("/api/chat", json={"message": "Test thinking privacy"})
    assert response.status_code == 200
    data = response.json()

    # Verify API response does not expose thinking
    assert data["thinking"] is None
    assert "Internal prompt analysis" not in response.text
    assert data["reply"] == "AI generated memory response"
    # Verify token usage is preserved
    assert data["usage"]["completion_tokens"] == 467

    # Verify conversation history in database only stores final reply
    conv_id = data["conversation_id"]
    messages = await memory_service.get_messages(conv_id)
    assert len(messages) == 2
    assistant_msg = messages[1]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "AI generated memory response"
    assert "Internal prompt analysis" not in assistant_msg.content


@pytest.mark.asyncio
async def test_chat_stream_raw_thinking_not_streamed_or_stored(
    api_client: tuple[AsyncClient, MockMemoryLLMProvider, MemoryService],
) -> None:
    """Verify streaming suppresses raw thinking, emits thinking indicator, and stores only clean reply."""
    client, mock_provider, memory_service = api_client
    mock_provider.stream_thinking_to_inject = [
        "Hidden thought token 1",
        "Hidden thought token 2",
    ]

    response = await client.post("/api/chat/stream", json={"message": "Stream privacy test"})
    assert response.status_code == 200
    body = response.text

    # Raw thinking tokens must never appear in stream output
    assert "Hidden thought token 1" not in body
    assert "Hidden thought token 2" not in body

    # Sanitized indicator must be present
    assert "AARYX is thinking..." in body

    # Actual content must be streamed
    assert "Streaming " in body
    assert "response content." in body

    # Extract conversation_id from stream chunks
    import json

    first_line = next(line for line in body.split("\n") if line.startswith("data: {"))
    chunk_json = json.loads(first_line.removeprefix("data: "))
    conv_id = chunk_json["conversation_id"]

    # Verify database contents: strictly final text, zero thinking
    messages = await memory_service.get_messages(conv_id)
    assert len(messages) == 2
    assistant_msg = messages[1]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "Streaming response content."
    assert "Hidden thought" not in assistant_msg.content
    assert "AARYX is thinking..." not in assistant_msg.content
