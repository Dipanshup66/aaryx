"""Regression tests for failed / empty / partially completed chat turns.

Covers review finding #4. Turn policy under test:

* The user's message is persisted before inference and is KEPT if the turn fails
  (existing contract, see test_chat_stream_interrupted_does_not_save_assistant_reply).
* No assistant message is ever stored for a failed, empty or thinking-only reply.
* A generation that succeeded but produced no visible text is not an HTTP error.
* A streaming turn only reports completion (`[DONE]`) after generation AND persistence succeed.
* The prompt sent to the provider never contains consecutive same-role messages built from
  stored history, so an unanswered earlier user message cannot wedge the conversation.
"""

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.deps import get_memory_service, get_orchestrator, get_provider
from app.brain.orchestrator import BrainOrchestrator, merge_unanswered_turns
from app.brain.schemas import BrainChatRequest
from app.main import app
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.service import MemoryService
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
)


class ScriptedProvider(LLMProvider):
    """Provider whose generate/stream behaviour is scripted per test."""

    def __init__(self) -> None:
        self.reply = "A normal reply"
        self.stream_deltas: list[str] = ["A normal ", "reply"]
        self.generate_error: Exception | None = None
        self.stream_error_after: int | None = None
        self.roles_seen: list[list[str]] = []
        self.contents_seen: list[list[str]] = []

    def _record(self, request: ChatRequest) -> None:
        self.roles_seen.append([m.role for m in request.messages])
        self.contents_seen.append([m.content for m in request.messages])

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self._record(request)
        if self.generate_error:
            raise self.generate_error
        return ChatResponse(content=self.reply, model="m", finish_reason="stop", provider="p")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self._record(request)
        for i, delta in enumerate(self.stream_deltas):
            if self.stream_error_after is not None and i == self.stream_error_after:
                raise ProviderResponseError("stream dropped", provider="p", status_code=502)
            last = i == len(self.stream_deltas) - 1
            yield ChatStreamChunk(
                delta=delta, model="m", provider="p", finish_reason="stop" if last else None
            )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="p")

    async def aclose(self) -> None:
        pass


@pytest_asyncio.fixture
async def env(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, ScriptedProvider, MemoryService]]:
    engine: AsyncEngine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'turns.db'}")
    await init_db(engine)
    service = MemoryService(session_factory=get_session_factory(engine), max_history_messages=10)
    provider = ScriptedProvider()
    orchestrator = BrainOrchestrator(provider=provider, max_history_messages=10)

    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_memory_service] = lambda: service
    app.state.provider = provider
    app.state.orchestrator = orchestrator
    app.state.memory_service = service
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, provider, service
    finally:
        app.dependency_overrides.clear()
        app.state.memory_service = None
        await engine.dispose()


async def _asgi_stream(path: str, body: dict[str, Any]) -> tuple[str, BaseException | None]:
    """Drive the ASGI app directly so partial SSE output survives an exception."""
    payload = json.dumps(body).encode()
    sent: list[dict[str, Any]] = []
    delivered = False
    finished = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        # Like a real client: stay connected until the response is complete.
        await finished.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            finished.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "server": ("test", 80),
        "client": ("test", 1),
        "scheme": "http",
    }
    error: BaseException | None = None
    try:
        await app(scope, receive, send)
    except BaseException as exc:  # noqa: BLE001 - test helper must capture everything
        error = exc
    finally:
        finished.set()
    text = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return text.decode(), error


def _conv_id_from_sse(body: str) -> str:
    line = next(ln for ln in body.split("\n") if ln.startswith("data: {"))
    return json.loads(line.removeprefix("data: "))["conversation_id"]


async def _only_conversation_id(service: MemoryService) -> str:
    page = await service.list_conversations()
    assert page.total == 1
    return page.conversations[0].id


# --------------------------------------------------------------------------- /chat


@pytest.mark.parametrize(
    "reply", ["<think>only reasoning</think>", "", "   \n", "<think>x</think> "]
)
async def test_chat_empty_or_thinking_only_reply_is_not_an_http_error(env, reply: str) -> None:
    client, provider, service = env
    provider.reply = reply

    res = await client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200
    assert res.json()["reply"] == ""
    conv_id = res.json()["conversation_id"]
    msgs = await service.get_messages(conv_id)
    assert [m.role for m in msgs] == ["user"]  # user kept, no empty assistant message


async def test_chat_provider_failure_keeps_user_message_and_stores_no_assistant(env) -> None:
    client, provider, service = env
    provider.generate_error = ProviderConnectionError("down", provider="p")

    res = await client.post("/api/chat", json={"message": "will fail"})

    assert res.status_code == 503
    conv_id = await _only_conversation_id(service)
    msgs = await service.get_messages(conv_id)
    assert [(m.role, m.content) for m in msgs] == [("user", "will fail")]


async def test_next_turn_after_failure_has_no_consecutive_user_roles(env) -> None:
    client, provider, service = env
    provider.generate_error = ProviderConnectionError("down", provider="p")
    await client.post("/api/chat", json={"message": "first, fails"})
    conv_id = await _only_conversation_id(service)

    provider.generate_error = None
    res = await client.post("/api/chat", json={"conversation_id": conv_id, "message": "second"})

    assert res.status_code == 200
    roles = provider.roles_seen[-1]
    assert all(a != b for a, b in itertools.pairwise(roles)), roles
    assert roles == ["system", "user"]
    # Nothing is lost: the failed message text is merged into the same user turn.
    sent_user_text = provider.contents_seen[-1][1]
    assert "first, fails" in sent_user_text and "second" in sent_user_text
    # Stored history keeps both user messages and the new assistant reply.
    stored = await service.get_messages(conv_id)
    assert [m.role for m in stored] == ["user", "user", "assistant"]


async def test_stream_next_turn_after_failed_stream_has_alternating_roles(env) -> None:
    _, provider, service = env
    provider.stream_error_after = 1
    _, error = await _asgi_stream("/api/chat/stream", {"message": "dies midway"})
    assert error is not None
    conv_id = await _only_conversation_id(service)

    provider.stream_error_after = None
    body, error = await _asgi_stream(
        "/api/chat/stream", {"conversation_id": conv_id, "message": "retry"}
    )

    assert error is None
    assert "data: [DONE]" in body
    assert provider.roles_seen[-1] == ["system", "user"]


# ----------------------------------------------------------------------- /chat/stream


async def test_stream_whitespace_only_reply_does_not_error_after_done(env) -> None:
    _, provider, service = env
    provider.stream_deltas = ["   ", "\n"]

    body, error = await _asgi_stream("/api/chat/stream", {"message": "hello"})

    assert error is None  # previously: ConversationValidationError escaped after [DONE]
    assert "data: [DONE]" in body
    conv_id = await _only_conversation_id(service)
    assert [m.role for m in await service.get_messages(conv_id)] == ["user"]


async def test_stream_thinking_only_reply_completes_without_storing_assistant(env) -> None:
    _, provider, service = env
    provider.stream_deltas = ["<think>reasoning", " only</think>"]

    body, error = await _asgi_stream("/api/chat/stream", {"message": "hello"})

    assert error is None
    assert "reasoning" not in body
    assert "data: [DONE]" in body
    conv_id = await _only_conversation_id(service)
    assert [m.role for m in await service.get_messages(conv_id)] == ["user"]


async def test_stream_persistence_failure_is_not_reported_as_success(env, monkeypatch) -> None:
    _, _, service = env

    async def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(service, "save_message", boom)

    body, error = await _asgi_stream("/api/chat/stream", {"message": "hello"})

    assert isinstance(error, RuntimeError)
    assert "data: [DONE]" not in body


async def test_stream_generation_failure_never_reports_done_and_keeps_user_message(env) -> None:
    _, provider, service = env
    provider.stream_error_after = 1

    body, error = await _asgi_stream("/api/chat/stream", {"message": "partial"})

    # Starlette re-raises post-header failures as the original error or a RuntimeError wrapper.
    assert isinstance(error, (ProviderResponseError, RuntimeError))
    assert "data: [DONE]" not in body
    assert "A normal " in body  # the partial text the client already received
    conv_id = await _only_conversation_id(service)
    msgs = await service.get_messages(conv_id)
    assert [(m.role, m.content) for m in msgs] == [("user", "partial")]


async def test_stream_success_persists_sanitized_reply_before_done(env) -> None:
    _, provider, service = env
    provider.stream_deltas = ["<think>x</think>Hel", "lo"]

    body, error = await _asgi_stream("/api/chat/stream", {"message": "hi"})

    assert error is None
    assert body.rstrip().endswith("data: [DONE]")
    conv_id = _conv_id_from_sse(body)
    msgs = await service.get_messages(conv_id)
    assert [(m.role, m.content) for m in msgs] == [("user", "hi"), ("assistant", "Hello")]


# ------------------------------------------------------------- merge_unanswered_turns


def test_merge_folds_orphaned_trailing_user_message_into_prompt() -> None:
    history, message = merge_unanswered_turns(
        [ChatMessage(role="user", content="unanswered")], "now"
    )
    assert history == []
    assert message == "unanswered\n\nnow"


def test_merge_folds_several_trailing_user_messages_in_order() -> None:
    history = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="failed 1"),
        ChatMessage(role="user", content="failed 2"),
    ]
    out, message = merge_unanswered_turns(history, "now")
    assert [m.role for m in out] == ["user", "assistant"]
    assert message == "failed 1\n\nfailed 2\n\nnow"


def test_merge_merges_orphan_in_the_middle_of_history() -> None:
    history = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="q2 failed"),
        ChatMessage(role="user", content="q3"),
        ChatMessage(role="assistant", content="a3"),
    ]
    out, message = merge_unanswered_turns(history, "now")
    assert [m.role for m in out] == ["user", "assistant", "user", "assistant"]
    assert out[2].content == "q2 failed\n\nq3"
    assert message == "now"


def test_merge_leaves_well_formed_history_untouched_and_does_not_mutate_input() -> None:
    history = [
        ChatMessage(role="user", content="q1"),
        ChatMessage(role="assistant", content="a1"),
    ]
    snapshot = [m.model_copy() for m in history]
    out, message = merge_unanswered_turns(history, "now")
    assert out == snapshot
    assert message == "now"
    assert history == snapshot


def test_caller_supplied_history_is_not_rewritten_by_the_orchestrator() -> None:
    """The orchestrator itself stays a pass-through (see test_context_message_limit_enforced)."""
    orch = BrainOrchestrator(provider=ScriptedProvider(), max_history_messages=20)
    out = orch._assemble_messages(
        BrainChatRequest(message="now", history=[ChatMessage(role="user", content="x")])
    )
    assert [m.role for m in out] == ["system", "user", "user"]
