"""Unit tests for BrainOrchestrator using a FakeLLMProvider."""

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from app.brain.orchestrator import BrainOrchestrator
from app.brain.prompts import get_system_prompt
from app.brain.schemas import BrainChatRequest
from app.providers.base import (
    ChatMessage,
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


class FakeLLMProvider(LLMProvider):
    """Controllable test double for LLMProvider."""

    def __init__(self, response_text: str = "AARYX response") -> None:
        self.response_text = response_text
        self.last_request: ChatRequest | None = None
        self.simulate_error: Exception | None = None
        self.is_closed = False

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.last_request = request
        if self.simulate_error:
            raise self.simulate_error
        return ChatResponse(
            content=self.response_text,
            model=request.model or "fake-model:4b",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.last_request = request
        if self.simulate_error:
            raise self.simulate_error
        words = self.response_text.split(" ")
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield ChatStreamChunk(
                delta=word + (" " if not is_last else ""),
                model=request.model or "fake-model:4b",
                finish_reason="stop" if is_last else None,
                usage=TokenUsage(prompt_tokens=15, completion_tokens=10, total_tokens=25)
                if is_last
                else None,
            )

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name="fake-model:4b")]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="fake")

    async def aclose(self) -> None:
        self.is_closed = True


@pytest.fixture
def fake_provider() -> FakeLLMProvider:
    return FakeLLMProvider(response_text="Hello! I am AARYX, your companion.")


@pytest.fixture
def orchestrator(fake_provider) -> BrainOrchestrator:
    return BrainOrchestrator(
        provider=fake_provider, max_history_messages=10, default_temperature=0.7
    )


@pytest.mark.asyncio
async def test_system_persona_inclusion(orchestrator, fake_provider):
    """Verify system persona is injected as the first prompt message."""
    request = BrainChatRequest(message="Hi AARYX!")
    await orchestrator.process(request)

    assert fake_provider.last_request is not None
    messages = fake_provider.last_request.messages
    assert len(messages) == 2  # system + current user message

    system_msg = messages[0]
    assert system_msg.role == "system"
    assert "AARYX" in system_msg.content
    assert "More Than AI. Your Intelligent Companion." in system_msg.content
    assert system_msg.content == get_system_prompt()


@pytest.mark.asyncio
async def test_message_order_preserved(orchestrator, fake_provider):
    """Verify chronological order: system -> history -> current message."""
    history = [
        ChatMessage(role="user", content="Turn 1 User"),
        ChatMessage(role="assistant", content="Turn 1 Assistant"),
    ]
    request = BrainChatRequest(message="Turn 2 User", history=history)
    await orchestrator.process(request)

    messages = fake_provider.last_request.messages
    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[1].content == "Turn 1 User"
    assert messages[1].role == "user"
    assert messages[2].content == "Turn 1 Assistant"
    assert messages[2].role == "assistant"
    assert messages[3].content == "Turn 2 User"
    assert messages[3].role == "user"


@pytest.mark.asyncio
async def test_successful_response(orchestrator):
    """Verify complete response mapping and usage metadata."""
    request = BrainChatRequest(message="What is the plan?")
    response = await orchestrator.process(request)

    assert response.reply == "Hello! I am AARYX, your companion."
    assert response.model == "fake-model:4b"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 15
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 25


@pytest.mark.asyncio
async def test_streaming_response(orchestrator):
    """Verify token streaming and assembly from provider."""
    request = BrainChatRequest(message="Stream this message")
    chunks = []
    final_chunk = None

    async for chunk in orchestrator.process_stream(request):
        chunks.append(chunk.delta)
        if chunk.finish_reason is not None:
            final_chunk = chunk

    full_output = "".join(chunks)
    assert full_output == "Hello! I am AARYX, your companion."
    assert final_chunk is not None
    assert final_chunk.finish_reason == "stop"
    assert final_chunk.usage is not None
    assert final_chunk.usage.total_tokens == 25


def test_empty_user_message_rejected():
    """Verify validation error when user message is empty or whitespace."""
    with pytest.raises(ValidationError) as exc_info:
        BrainChatRequest(message="")
    assert "User message cannot be empty" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        BrainChatRequest(message="   \n\t  ")
    assert "User message cannot be empty" in str(exc_info.value)


def test_invalid_conversation_role_rejected():
    """Verify invalid conversation roles are rejected by schema."""
    with pytest.raises(ValidationError) as exc_info:
        ChatMessage(role="superuser", content="Invalid message")  # type: ignore[arg-type]
    assert "role" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_connection_error_propagated(orchestrator, fake_provider):
    """Verify ProviderConnectionError is raised cleanly when daemon is unreachable."""
    fake_provider.simulate_error = ProviderConnectionError("Cannot reach daemon", provider="ollama")

    with pytest.raises(ProviderConnectionError) as exc_info:
        await orchestrator.process(BrainChatRequest(message="Hello"))
    assert "Cannot reach daemon" in str(exc_info.value)
    assert exc_info.value.provider == "ollama"


@pytest.mark.asyncio
async def test_provider_timeout_error_propagated(orchestrator, fake_provider):
    """Verify ProviderTimeoutError is raised cleanly on generation timeout."""
    fake_provider.simulate_error = ProviderTimeoutError("Request timed out", provider="ollama")

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await orchestrator.process(BrainChatRequest(message="Hello"))
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_response_error_propagated(orchestrator, fake_provider):
    """Verify ProviderResponseError is raised on 5xx or invalid responses."""
    fake_provider.simulate_error = ProviderResponseError("Ollama returned 500", provider="ollama")

    with pytest.raises(ProviderResponseError) as exc_info:
        await orchestrator.process(BrainChatRequest(message="Hello"))
    assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_context_message_limit_enforced(fake_provider):
    """Verify orchestrator bounds conversation context to max_history_messages."""
    # Set limit to 3 history messages
    orchestrator = BrainOrchestrator(provider=fake_provider, max_history_messages=3)

    history = [
        ChatMessage(role="user", content="Msg 1"),
        ChatMessage(role="assistant", content="Msg 2"),
        ChatMessage(role="user", content="Msg 3"),
        ChatMessage(role="assistant", content="Msg 4"),
        ChatMessage(role="user", content="Msg 5"),
    ]
    request = BrainChatRequest(message="Msg 6 (Current)", history=history)
    await orchestrator.process(request)

    messages = fake_provider.last_request.messages
    # 1 system + 3 history + 1 current = 5 total
    assert len(messages) == 5
    assert messages[0].role == "system"
    assert messages[1].content == "Msg 3"
    assert messages[2].content == "Msg 4"
    assert messages[3].content == "Msg 5"
    assert messages[4].content == "Msg 6 (Current)"


@pytest.mark.asyncio
async def test_generation_configuration_passed(orchestrator, fake_provider):
    """Verify model overrides, temperature, and token budgets are passed to provider."""
    request = BrainChatRequest(
        message="Configure me",
        model="qwen3:4b",
        temperature=0.2,
        max_tokens=512,
    )
    await orchestrator.process(request)

    req = fake_provider.last_request
    assert req is not None
    assert req.model == "qwen3:4b"
    assert req.temperature == 0.2
    assert req.max_tokens == 512


@pytest.mark.asyncio
async def test_raw_thinking_not_exposed_in_non_streaming():
    """Verify raw thinking field from provider is strictly blocked from BrainChatResponse."""

    class ThinkingProvider(FakeLLMProvider):
        async def generate(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                content="Final clean answer.",
                model="qwen3.5:4b",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=467, total_tokens=477),
                thinking="Step-by-step secret chain of thought that must not be exposed.",
                provider="ollama",
            )

    provider = ThinkingProvider()
    orchestrator = BrainOrchestrator(provider=provider)

    response = await orchestrator.process(BrainChatRequest(message="Test"))

    assert response.reply == "Final clean answer."
    assert response.thinking is None
    dumped = response.model_dump()
    assert dumped["thinking"] is None
    assert "secret chain of thought" not in str(dumped)
    # Token usage must still be preserved accurately
    assert response.usage is not None
    assert response.usage.completion_tokens == 467


@pytest.mark.asyncio
async def test_embedded_thinking_tags_stripped_from_content():
    """Verify <think>...</think> tags embedded inside content are sanitized out."""

    class EmbeddedTagProvider(FakeLLMProvider):
        async def generate(self, request: ChatRequest) -> ChatResponse:
            return ChatResponse(
                content="<think>\nI am planning the answer\n</think>\nHere is the real answer.",
                model="qwen3.5:4b",
                finish_reason="stop",
            )

    provider = EmbeddedTagProvider()
    orchestrator = BrainOrchestrator(provider=provider)

    response = await orchestrator.process(BrainChatRequest(message="Test tags"))
    assert response.reply == "Here is the real answer."
    assert "<think>" not in response.reply
    assert "planning the answer" not in response.reply


@pytest.mark.asyncio
async def test_streaming_thinking_sanitization():
    """Verify streaming suppresses raw thinking tokens and emits a clean indicator."""

    class StreamingThinkingProvider(FakeLLMProvider):
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            # Emits 2 thinking chunks followed by 2 text chunks
            yield ChatStreamChunk(delta="", thinking="Raw internal reasoning chunk 1")
            yield ChatStreamChunk(delta="", thinking="Raw internal reasoning chunk 2")
            yield ChatStreamChunk(delta="Hello ")
            yield ChatStreamChunk(delta="world!", finish_reason="stop")

    provider = StreamingThinkingProvider()
    orchestrator = BrainOrchestrator(provider=provider)

    chunks = []
    async for chunk in orchestrator.process_stream(BrainChatRequest(message="Stream thinking")):
        chunks.append(chunk)

    # Verify that raw reasoning tokens never leaked
    all_chunks_str = str([c.model_dump() for c in chunks])
    assert "Raw internal reasoning chunk 1" not in all_chunks_str
    assert "Raw internal reasoning chunk 2" not in all_chunks_str

    # Exactly one indicator chunk should have been yielded
    indicator_chunks = [c for c in chunks if c.thinking == "AARYX is thinking..."]
    assert len(indicator_chunks) == 1
    assert indicator_chunks[0].delta == ""

    # Content deltas must remain clean
    content_text = "".join(c.delta for c in chunks)
    assert content_text == "Hello world!"


@pytest.mark.asyncio
async def test_streaming_embedded_think_tags_filtered():
    """Verify streaming filters embedded <think>...</think> tags appearing in delta stream."""

    class StreamingEmbeddedTagProvider(FakeLLMProvider):
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(delta="<think>Hidden reasoning ")
            yield ChatStreamChunk(delta="continuation</think>Visible text")

    provider = StreamingEmbeddedTagProvider()
    orchestrator = BrainOrchestrator(provider=provider)

    chunks = []
    async for chunk in orchestrator.process_stream(
        BrainChatRequest(message="Stream embedded tags")
    ):
        chunks.append(chunk)

    content_text = "".join(c.delta for c in chunks)
    assert content_text == "Visible text"
    assert "Hidden reasoning" not in content_text
