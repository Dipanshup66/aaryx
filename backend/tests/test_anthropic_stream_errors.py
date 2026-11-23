"""Regression tests for Anthropic streaming error handling (review finding #8).

* In-stream ``error`` events (overloaded, rate limit, ...) must raise, not end quietly.
* A stream that ends without a terminal event is truncated and must raise.
* Non-timeout httpx failures must be normalized into ProviderError subclasses.
"""

import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ProviderConnectionError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.providers.router import is_transient_error

REQUEST = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model="claude-test")


def _sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)


TEXT = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "partial "}}
FINISH = {
    "type": "message_delta",
    "delta": {"stop_reason": "end_turn"},
    "usage": {"output_tokens": 3},
}
STOP = {"type": "message_stop"}


def _error(error_type: str, message: str = "boom") -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _provider(handler: Callable[[httpx.Request], httpx.Response]) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def _drain(provider: AnthropicProvider) -> list[str]:
    return [c.delta async for c in provider.stream(REQUEST)]


class _BrokenStream(httpx.AsyncByteStream):
    """Yields some bytes, then fails like a dropped connection."""

    def __init__(self, body: bytes, error: Exception) -> None:
        self.body = body
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.body
        raise self.error


# ------------------------------------------------------------------ in-stream error events


@pytest.mark.parametrize(
    ("error_type", "status", "transient"),
    [
        ("overloaded_error", 529, True),
        ("rate_limit_error", 429, True),
        ("api_error", 500, True),
        ("invalid_request_error", 400, False),
        ("authentication_error", 401, False),
        ("permission_error", 403, False),
        ("not_found_error", 404, False),
        ("request_too_large", 413, False),
    ],
)
async def test_error_event_mid_stream_raises_with_mapped_status(
    error_type: str, status: int, transient: bool
) -> None:
    body = _sse(TEXT, _error(error_type, "something went wrong"))
    provider = _provider(lambda request: httpx.Response(200, content=body))

    received: list[str] = []
    with pytest.raises(ProviderResponseError) as info:
        async for chunk in provider.stream(REQUEST):
            received.append(chunk.delta)

    assert received == ["partial "]  # text before the error was delivered
    assert info.value.status_code == status
    assert info.value.provider == "anthropic"
    assert "something went wrong" in str(info.value)
    assert is_transient_error(info.value) is transient


async def test_error_event_before_any_text_raises() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=_sse(_error("overloaded_error", "Overloaded")))
    )
    with pytest.raises(ProviderResponseError) as info:
        await _drain(provider)
    assert info.value.status_code == 529


async def test_unknown_error_event_type_raises_without_status() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=_sse(TEXT, _error("brand_new_error")))
    )
    with pytest.raises(ProviderResponseError) as info:
        await _drain(provider)
    assert info.value.status_code is None


async def test_error_event_secrets_are_not_exposed_via_safe_message() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=_sse(_error("authentication_error", "bad key sk-ant-ABCDEFGHIJKLMNOP")),
        )
    )
    with pytest.raises(ProviderResponseError) as info:
        await _drain(provider)
    assert "ABCDEFGHIJKLMNOP" not in info.value.safe_message


# ------------------------------------------------------------------------ truncated streams


async def test_stream_without_terminal_event_is_truncated_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=_sse(TEXT)))
    received: list[str] = []
    with pytest.raises(ProviderResponseError):
        async for chunk in provider.stream(REQUEST):
            received.append(chunk.delta)
    assert received == ["partial "]


async def test_empty_stream_is_truncated_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b""))
    with pytest.raises(ProviderResponseError):
        await _drain(provider)


async def test_complete_stream_still_succeeds() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=_sse(TEXT, FINISH, STOP)))
    chunks = [c async for c in provider.stream(REQUEST)]
    assert [c.delta for c in chunks] == ["partial ", ""]
    assert chunks[-1].finish_reason == "end_turn"


async def test_stream_with_stop_reason_but_no_message_stop_is_truncated() -> None:
    """Exact scenario: message_start, text deltas, message_delta(stop_reason), connection closes.

    Anthropic's documented flow ends with a FINAL message_stop, and a stop_reason arrives on
    one of "one or more" message_delta events, so it is not proof of completion.
    """
    body = _sse(
        {"type": "message_start", "message": {"id": "m1", "usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        TEXT,
        FINISH,
        # connection closes: no content_block_stop, no message_stop
    )
    provider = _provider(lambda request: httpx.Response(200, content=body))

    received: list[str] = []
    with pytest.raises(ProviderResponseError) as info:
        async for chunk in provider.stream(REQUEST):
            received.append(chunk.delta)

    assert received == ["partial ", ""]  # what was already delivered stays delivered
    assert "message_stop" in str(info.value)
    assert info.value.provider == "anthropic"


async def test_error_event_after_stop_reason_is_still_reported() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200, content=_sse(TEXT, FINISH, _error("overloaded_error", "Overloaded"))
        )
    )
    with pytest.raises(ProviderResponseError) as info:
        await _drain(provider)
    assert info.value.status_code == 529


async def test_multiple_message_deltas_then_message_stop_succeeds() -> None:
    second = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 9},
    }
    provider = _provider(
        lambda request: httpx.Response(200, content=_sse(TEXT, FINISH, second, STOP))
    )
    chunks = [c async for c in provider.stream(REQUEST)]
    assert chunks[-1].usage.completion_tokens == 9


async def test_truncated_anthropic_stream_is_not_saved_as_assistant_reply(tmp_path) -> None:
    """End to end: provider -> orchestrator -> /api/chat/stream -> database."""
    from app.api.deps import get_memory_service, get_orchestrator, get_provider
    from app.brain.orchestrator import BrainOrchestrator
    from app.main import app
    from app.memory.database import create_database_engine, get_session_factory, init_db
    from app.memory.service import MemoryService

    body = _sse(
        {"type": "message_start", "message": {"id": "m1"}},
        TEXT,
        FINISH,
    )
    provider = _provider(lambda request: httpx.Response(200, content=body))
    engine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'trunc.db'}")
    await init_db(engine)
    service = MemoryService(session_factory=get_session_factory(engine))
    orchestrator = BrainOrchestrator(provider=provider)

    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_memory_service] = lambda: service
    app.state.provider, app.state.orchestrator, app.state.memory_service = (
        provider,
        orchestrator,
        service,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            with pytest.raises((ProviderResponseError, RuntimeError)):
                await client.post("/api/chat/stream", json={"message": "hello there"})

        page = await service.list_conversations()
        assert page.total == 1
        stored = await service.get_messages(page.conversations[0].id)
        assert [(m.role, m.content) for m in stored] == [("user", "hello there")]
    finally:
        app.dependency_overrides.clear()
        app.state.memory_service = None
        await engine.dispose()


async def test_ping_and_unrelated_events_do_not_mask_truncation() -> None:
    body = _sse({"type": "ping"}, {"type": "message_start", "message": {}}, TEXT)
    provider = _provider(lambda request: httpx.Response(200, content=body))
    with pytest.raises(ProviderResponseError):
        await _drain(provider)


# ------------------------------------------------------------- httpx exception normalization


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("peer closed connection without sending complete message"),
        httpx.ReadError("connection reset"),
        httpx.WriteError("broken pipe"),
        httpx.ConnectError("refused"),
        httpx.CloseError("closed"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_stream_transport_errors_become_connection_errors(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ProviderConnectionError) as info:
        await _drain(_provider(handler))
    assert info.value.provider == "anthropic"
    assert is_transient_error(info.value) is True


@pytest.mark.parametrize(
    "error",
    [
        httpx.RemoteProtocolError("incomplete chunked read"),
        httpx.ReadError("connection reset by peer"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_mid_stream_transport_failure_becomes_connection_error(error: Exception) -> None:
    body = _sse(TEXT)
    provider = _provider(lambda request: httpx.Response(200, stream=_BrokenStream(body, error)))
    received: list[str] = []
    with pytest.raises(ProviderConnectionError):
        async for chunk in provider.stream(REQUEST):
            received.append(chunk.delta)
    assert received == ["partial "]


async def test_mid_stream_read_timeout_is_timeout_error() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200, stream=_BrokenStream(_sse(TEXT), httpx.ReadTimeout("read timed out"))
        )
    )
    with pytest.raises(ProviderTimeoutError):
        await _drain(provider)


async def test_mid_stream_decoding_error_is_provider_response_error() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200, stream=_BrokenStream(_sse(TEXT), httpx.DecodingError("bad gzip"))
        )
    )
    with pytest.raises(ProviderResponseError):
        await _drain(provider)


async def test_no_raw_httpx_exception_escapes_stream() -> None:
    for error in (
        httpx.RemoteProtocolError("x"),
        httpx.ReadError("x"),
        httpx.DecodingError("x"),
        httpx.TooManyRedirects("x"),
        httpx.UnsupportedProtocol("x"),
    ):

        def handler(request: httpx.Request, error: Exception = error) -> httpx.Response:
            raise error

        with pytest.raises(ProviderError):
            await _drain(_provider(handler))


@pytest.mark.parametrize(
    "error",
    [httpx.RemoteProtocolError("peer closed"), httpx.ReadError("reset"), httpx.WriteError("pipe")],
    ids=lambda e: type(e).__name__,
)
async def test_generate_transport_errors_become_connection_errors(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(ProviderConnectionError):
        await _provider(handler).generate(REQUEST)


async def test_generate_timeout_and_http_status_unchanged() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeoutError):
        await _provider(timeout_handler).generate(REQUEST)

    with pytest.raises(ProviderResponseError) as info:
        await _provider(lambda r: httpx.Response(503, text="down")).generate(REQUEST)
    assert info.value.status_code == 503
