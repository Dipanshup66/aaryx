"""latency_ms (P1-12): every reply and every finished stream says how long the model took."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import app.brain.orchestrator as orchestrator_module
from app.brain.orchestrator import BrainOrchestrator
from app.brain.schemas import BrainChatRequest, BrainChatResponse, BrainStreamChunk
from app.core.config import Settings
from app.main import create_app
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConnectionError,
    ProviderHealth,
)


class TimedProvider(LLMProvider):
    """Model double that takes a configurable time per step."""

    def __init__(
        self, delay: float = 0.0, steps: tuple[str, ...] = ("Hel", "lo ", "there")
    ) -> None:
        self.delay = delay
        self.steps = steps
        self.finish = True
        self.error: Exception | None = None

    async def generate(self, request: ChatRequest) -> ChatResponse:
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return ChatResponse(content="Hello there", model="m", provider="timed")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        if self.error:
            raise self.error
        for i, piece in enumerate(self.steps):
            await asyncio.sleep(self.delay)
            last = i == len(self.steps) - 1
            yield ChatStreamChunk(
                delta=piece,
                model="m",
                provider="timed",
                finish_reason="stop" if last and self.finish else None,
            )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="timed")

    async def aclose(self) -> None:
        pass


def _request(text: str = "hello") -> BrainChatRequest:
    return BrainChatRequest(message=text)


async def _collect(provider: TimedProvider) -> list[BrainStreamChunk]:
    return [c async for c in BrainOrchestrator(provider=provider).process_stream(_request())]


# ===================================================================== non-streaming reply


async def test_reply_reports_how_long_the_model_took() -> None:
    response = await BrainOrchestrator(provider=TimedProvider(delay=0.06)).process(_request())

    assert isinstance(response.latency_ms, int) and not isinstance(response.latency_ms, bool)
    assert 50 <= response.latency_ms < 3000


async def test_an_instant_model_reports_a_small_non_negative_number() -> None:
    response = await BrainOrchestrator(provider=TimedProvider()).process(_request())

    assert 0 <= response.latency_ms < 500


async def test_latency_covers_everything_the_provider_did_including_internal_retries() -> None:
    class RetryingProvider(TimedProvider):
        async def generate(self, request: ChatRequest) -> ChatResponse:
            await asyncio.sleep(0.04)  # first attempt (fails in real life)
            return await super().generate(request)

    provider = RetryingProvider(delay=0.04)

    response = await BrainOrchestrator(provider=provider).process(_request())

    assert response.latency_ms >= 70


async def test_a_failed_reply_has_no_latency_to_report() -> None:
    provider = TimedProvider()
    provider.error = ProviderConnectionError("down", provider="timed")

    with pytest.raises(ProviderConnectionError):
        await BrainOrchestrator(provider=provider).process(_request())


@pytest.mark.parametrize(
    ("ticks", "expected"),
    [
        ((10.0, 10.2503), 250),
        ((10.0, 10.2507), 251),
        ((10.0, 10.0), 0),
        ((10.0, 9.0), 0),  # a clock that goes backwards never yields a negative number
        ((0.0, 61.0), 61000),
    ],
)
async def test_milliseconds_are_whole_rounded_and_never_negative(
    monkeypatch: pytest.MonkeyPatch, ticks: tuple[float, float], expected: int
) -> None:
    readings = iter(ticks)
    monkeypatch.setattr(orchestrator_module.time, "perf_counter", lambda: next(readings))

    response = await BrainOrchestrator(provider=TimedProvider()).process(_request())

    assert response.latency_ms == expected


def test_schema_accepts_old_style_construction_and_rejects_negative_values() -> None:
    assert BrainChatResponse(reply="x", model="m").latency_ms is None
    assert BrainStreamChunk(delta="x").latency_ms is None
    with pytest.raises(ValidationError):
        BrainChatResponse(reply="x", model="m", latency_ms=-1)
    with pytest.raises(ValidationError):
        BrainStreamChunk(delta="x", latency_ms=-5)


# ======================================================================== streaming


async def test_only_the_last_chunk_of_a_stream_carries_the_latency() -> None:
    chunks = await _collect(TimedProvider(delay=0.03))

    with_latency = [c for c in chunks if c.latency_ms is not None]
    assert len(with_latency) == 1 and chunks[-1].latency_ms is not None
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].latency_ms >= 80, "three steps of 30 ms"
    assert all(c.latency_ms is None for c in chunks[:-1])


async def test_thinking_indicator_and_ordinary_chunks_carry_no_latency() -> None:
    class Thinks(TimedProvider):
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            yield ChatStreamChunk(delta="", model="m", provider="timed", thinking="hmm")
            yield ChatStreamChunk(delta="Hi", model="m", provider="timed", finish_reason="stop")

    chunks = await _collect(Thinks())

    assert chunks[0].thinking and chunks[0].latency_ms is None
    assert chunks[-1].latency_ms is not None


async def test_a_stream_without_a_finish_signal_puts_the_latency_on_the_released_tail() -> None:
    provider = TimedProvider(steps=("Hello <thi",))  # looks like the start of a reasoning tag
    provider.finish = False

    chunks = await _collect(provider)

    assert chunks[-1].delta == "<thi" and chunks[-1].latency_ms is not None


async def test_stream_latency_is_the_whole_duration_not_time_to_first_token() -> None:
    chunks = await _collect(TimedProvider(delay=0.05, steps=("a", "b", "c", "d")))

    assert chunks[-1].latency_ms >= 180


# ====================================================================== through the API


@asynccontextmanager
async def api_with(provider: LLMProvider):
    app = create_app(Settings(_env_file=None, memory_enabled=False))
    app.state.provider = provider
    app.state.orchestrator = BrainOrchestrator(provider=provider)
    app.state.memory_service = None
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://localhost") as client:
        yield client


async def test_chat_response_json_contains_latency_ms() -> None:
    async with api_with(TimedProvider(delay=0.06)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == 200
    assert isinstance(body["latency_ms"], int) and body["latency_ms"] >= 50
    assert {"reply", "model", "provider", "usage", "finish_reason"} <= set(body)  # unchanged


async def test_stream_events_carry_latency_only_on_the_final_chunk() -> None:
    async with api_with(TimedProvider(delay=0.03)) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: {")]
    final = [e for e in events if e["latency_ms"] is not None]
    assert len(final) == 1 and final[0]["finish_reason"] == "stop" and final[0]["latency_ms"] >= 80
    assert "data: [DONE]" in res.text


async def test_error_responses_do_not_pretend_to_have_a_latency() -> None:
    provider = TimedProvider()
    provider.error = ProviderConnectionError("down", provider="timed")

    async with api_with(provider) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 503 and "latency_ms" not in res.json()


# ============================================================================= logging


async def test_latency_is_logged_without_any_message_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="aaryx.brain.orchestrator")

    await BrainOrchestrator(provider=TimedProvider(delay=0.01)).process(
        _request("my secret question about Pune")
    )

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "latency_ms=" in text and "provider=timed" in text and "model=m" in text
    assert "secret" not in text and "Pune" not in text and "Hello there" not in text
