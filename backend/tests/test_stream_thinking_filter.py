"""Regression tests for the streaming reasoning-tag filter in BrainOrchestrator.process_stream.

Covers review finding #1: `<think>`/`<thought>` handling when tags arrive in one chunk,
are split across chunks, repeat, are malformed, or make up the entire response.
"""

from collections.abc import AsyncIterator

import pytest

from app.brain.orchestrator import BrainOrchestrator, sanitize_reply_content
from app.brain.schemas import BrainChatRequest, BrainStreamChunk
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
    TokenUsage,
)

USAGE = TokenUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)


class ScriptedStreamProvider(LLMProvider):
    """Provider that replays a fixed list of stream chunks."""

    def __init__(self, chunks: list[ChatStreamChunk]) -> None:
        self.chunks = chunks

    async def generate(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        for chunk in self.chunks:
            yield chunk

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="scripted")

    async def aclose(self) -> None:
        pass


def _chunks(deltas: list[str], *, finish_on_last: bool = True) -> list[ChatStreamChunk]:
    """Build provider chunks; the final one carries finish_reason and usage."""
    out: list[ChatStreamChunk] = []
    for i, delta in enumerate(deltas):
        last = i == len(deltas) - 1 and finish_on_last
        out.append(
            ChatStreamChunk(
                delta=delta,
                model="m",
                provider="scripted",
                finish_reason="stop" if last else None,
                usage=USAGE if last else None,
            )
        )
    return out


async def _run(chunks: list[ChatStreamChunk]) -> list[BrainStreamChunk]:
    orch = BrainOrchestrator(provider=ScriptedStreamProvider(chunks))
    request = BrainChatRequest(message="hi")
    return [c async for c in orch.process_stream(request)]


def _visible(out: list[BrainStreamChunk]) -> str:
    return "".join(c.delta for c in out)


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        # open and close inside a single chunk (the reported failure)
        (["<think>hmm</think>Hello"], "Hello"),
        (["<think>hmm</think>Hello", " world"], "Hello world"),
        (["Before <think>hmm</think>After"], "Before After"),
        # text before, between and after multiple blocks in one chunk
        (["A<think>x</think>B<think>y</think>C"], "ABC"),
        # <thought> variant, single chunk and multi chunk
        (["<thought>secret</thought>Visible"], "Visible"),
        (["<thought>sec", "ret</thought>Visible"], "Visible"),
        # tags split across chunk boundaries at every position
        (["<thi", "nk>secret", "</th", "ink>Visible"], "Visible"),
        (["<", "think", ">secret<", "/think", ">Visible"], "Visible"),
        (["<thou", "ght>secret</thou", "ght>Visible"], "Visible"),
        # case-insensitive tags (sanitize_reply_content is case-insensitive)
        (["<THINK>secret</Think>Visible"], "Visible"),
        # reasoning spans many chunks, then answer streams normally
        (["<think>", "a", "b", "c", "</think>", "Answer", " here"], "Answer here"),
        # no reasoning at all: untouched, including lone angle brackets
        (["Hello", " world"], "Hello world"),
        (["x <", " y"], "x < y"),
        (["1 < 2 and 3 > 2"], "1 < 2 and 3 > 2"),
        # partial tag prefix that never completes must not be swallowed at stream end
        (["hello <thi"], "hello <thi"),
        (["hello <"], "hello <"),
        # unclosed block: reasoning must never leak
        (["Hi <think>never closed"], "Hi "),
        (["Hi <think>never", " closed", " at all"], "Hi "),
        # closing tag without an opener is plain text (matches sanitize_reply_content)
        (["plain </think> text"], "plain </think> text"),
    ],
)
async def test_stream_filter_cases(deltas: list[str], expected: str) -> None:
    out = await _run(_chunks(deltas))
    visible = _visible(out)
    assert visible == expected
    assert "secret" not in visible
    # Final metadata must always be delivered exactly once.
    finals = [c for c in out if c.finish_reason]
    assert len(finals) == 1
    assert finals[0].finish_reason == "stop"
    assert finals[0].usage == USAGE


@pytest.mark.parametrize(
    "deltas",
    [
        ["<think>hmm</think>Hello"],
        ["A<think>x</think>B<think>y</think>C"],
        ["<thi", "nk>secret", "</th", "ink>Visible"],
        ["<thought>a</thought>b"],
        ["Hi <think>never closed"],
        ["x <", " y"],
        ["hello <thi"],
        ["plain </think> text"],
    ],
)
async def test_stream_output_matches_persisted_sanitization(deltas: list[str]) -> None:
    """What the client sees streamed equals what is persisted (sanitize of raw text)."""
    out = await _run(_chunks(deltas))
    assert _visible(out).strip() == sanitize_reply_content("".join(deltas))


async def test_thinking_only_response_emits_no_text_but_completes() -> None:
    out = await _run(_chunks(["<think>all reasoning, no answer</think>"]))
    assert _visible(out) == ""
    finals = [c for c in out if c.finish_reason]
    assert len(finals) == 1 and finals[0].usage == USAGE
    assert all("reasoning" not in (c.thinking or "") for c in out)


async def test_thinking_indicator_emitted_once_and_never_raw_reasoning() -> None:
    out = await _run(_chunks(["<think>a</think>x<think>b</think>y"]))
    indicators = [c for c in out if c.thinking]
    assert len(indicators) == 1
    assert indicators[0].thinking == "AARYX is thinking..."
    assert _visible(out) == "xy"


async def test_native_provider_thinking_is_suppressed() -> None:
    chunks = [
        ChatStreamChunk(delta="", model="m", thinking="raw chain of thought 1"),
        ChatStreamChunk(delta="", model="m", thinking="raw chain of thought 2"),
        ChatStreamChunk(delta="Answer", model="m", finish_reason="stop", usage=USAGE),
    ]
    out = await _run(chunks)
    assert _visible(out) == "Answer"
    assert [c.thinking for c in out if c.thinking] == ["AARYX is thinking..."]


async def test_delta_in_same_chunk_as_native_thinking_is_not_dropped() -> None:
    chunks = [
        ChatStreamChunk(delta="Visible", model="m", thinking="raw reasoning"),
        ChatStreamChunk(delta=" tail", model="m", finish_reason="stop", usage=USAGE),
    ]
    out = await _run(chunks)
    assert _visible(out) == "Visible tail"
    assert all("raw reasoning" not in (c.thinking or "") for c in out)


async def test_stream_without_finish_reason_still_flushes_held_text() -> None:
    out = await _run(_chunks(["done <"], finish_on_last=False))
    assert _visible(out) == "done <"
