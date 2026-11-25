"""Actionable errors and latency, attacked the way a real user and a real setup would:
many failure kinds, streaming and not, fallback and retries, memory on, parallel requests,
token auth, odd model names and hostile upstream bodies.

Everything runs the real Ollama / OpenAI adapters (over httpx.MockTransport), the real router,
the real orchestrator and the real FastAPI app.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from app.brain.orchestrator import BrainOrchestrator
from app.core.config import Settings
from app.main import create_app
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.extractor import InferredMemoryExtractor
from app.memory.service import MemoryService
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter

TOKEN = "scenario-token-" + "q" * 24
OLLAMA_OK = {
    "model": "qwen3.5:4b",
    "message": {"role": "assistant", "content": "Hello from the model"},
    "done": True,
    "done_reason": "stop",
}
OPENAI_OK = {
    "model": "gpt-4o-mini",
    "choices": [{"message": {"content": "Hello from the cloud"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
}


# ====================================================================================== harness


def ollama(handler, base_url: str = "http://mock-ollama:11434") -> OllamaProvider:
    return OllamaProvider(
        base_url=base_url,
        default_model="qwen3.5:4b",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def openai(handler) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="sk-test-key-for-tests-1234567890",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def make_router(
    primary: LLMProvider, cloud: LLMProvider | None = None, retries: int = 0
) -> ProviderRouter:
    settings = Settings(
        _env_file=None,
        ai_max_retries=retries,
        ai_fallback_enabled=cloud is not None,
        ai_fallback_provider="openai" if cloud else None,
        ai_allow_cloud_fallback=cloud is not None,
        openai_api_key="sk-test-key-for-tests-1234567890" if cloud else None,
    )
    router = ProviderRouter(settings=settings)
    router._provider_pool["ollama"] = primary
    if cloud is not None:
        router._provider_pool["openai"] = cloud
    return router


@asynccontextmanager
async def app_with(provider: LLMProvider, memory: MemoryService | None = None, **settings):
    app = create_app(Settings(_env_file=None, memory_enabled=memory is not None, **settings))
    app.state.provider = provider
    app.state.orchestrator = BrainOrchestrator(provider=provider)
    app.state.memory_service = memory
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://localhost") as client:
        yield client


def sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


class _NoExtraction(LLMProvider):
    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="[]", model="x", provider="ollama")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta="[]", model="x", provider="ollama")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="ollama")

    async def aclose(self) -> None:
        pass


@pytest_asyncio.fixture
async def memory(tmp_path: Path) -> AsyncIterator[MemoryService]:
    engine: AsyncEngine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'p1.db'}")
    await init_db(engine)
    service = MemoryService(
        session_factory=get_session_factory(engine), settings=Settings(_env_file=None)
    )
    service.extractor = InferredMemoryExtractor(settings=service.settings, provider=_NoExtraction())
    try:
        yield service
    finally:
        await engine.dispose()


# ======================================================== 1. every kind of failure x chat / stream


def down(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused")


def slow(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("took too long")


def missing(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"error": "model not found, try pulling it first"})


def status(code: int, text: str = "no") -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(code, text=text)


FAILURES = {
    "down": (down, 503, "provider_unreachable", "ollama serve"),
    "timeout": (slow, 504, "provider_timeout", "OLLAMA_TIMEOUT"),
    "missing-model": (missing, 502, "model_not_found", "ollama pull qwen3.5:4b"),
    "401": (status(401), 502, "authentication_failed", "OLLAMA_BASE_URL"),
    "403": (status(403), 502, "authentication_failed", "OLLAMA_BASE_URL"),
    "429": (status(429), 502, "rate_limited", "limiting requests"),
    "500": (status(500, "boom"), 502, "provider_response_error", None),
    "400": (
        status(400, '{"error":"context length exceeded"}'),
        502,
        "provider_response_error",
        None,
    ),
}


@pytest.mark.parametrize("kind", FAILURES)
async def test_1a_chat_failure_gives_the_right_status_code_and_hint(kind: str) -> None:
    handler, http_status, code, hint = FAILURES[kind]

    async with app_with(ollama(handler)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == http_status and body["code"] == code
    assert {"error", "detail", "provider"} <= set(body)
    if hint:
        assert hint in body["hint"]
    else:
        assert "hint" not in body


@pytest.mark.parametrize("kind", FAILURES)
async def test_1b_stream_failure_gives_the_same_payload_as_an_event(kind: str) -> None:
    handler, _, code, hint = FAILURES[kind]

    async with app_with(ollama(handler)) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    error = sse(res.text)[-1]["error"]
    assert error["code"] == code and "[DONE]" not in res.text
    if hint:
        assert hint in error["hint"]


async def test_1c_garbled_200_reply_is_a_clean_502_not_a_crash() -> None:
    async with app_with(
        ollama(lambda r: httpx.Response(200, text="<html>not json</html>"))
    ) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 502 and res.json()["code"] == "provider_response_error"


async def test_1d_stream_of_garbage_lines_ends_cleanly() -> None:
    async with app_with(ollama(lambda r: httpx.Response(200, text="garbage\n{broken\n"))) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    assert res.status_code == 200 and "[DONE]" in res.text
    assert not any("error" in e for e in sse(res.text))


# ============================================== 2. what an upstream error body can carry into a reply


async def test_2a_upstream_body_with_newlines_quotes_and_unicode_cannot_break_the_sse_stream() -> (
    None
):
    nasty = 'line one\n\ndata: {"fake": true}\r\n"quoted"   नमस्ते \x00 end'

    async with app_with(ollama(status(500, nasty))) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = sse(res.text)
    assert len(events) == 1 and "error" in events[0]
    assert "fake" not in events[0]  # the injected "event" never became an event
    assert all(not line.startswith('data: {"fake') for line in res.text.splitlines())


async def test_2b_a_huge_upstream_body_is_cut_to_a_bounded_detail() -> None:
    async with app_with(ollama(status(500, "x" * 200_000))) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert len(res.json()["detail"]) <= 600


async def test_2c_secrets_echoed_by_upstream_are_masked_in_chat_and_stream() -> None:
    key = "sk-leakedleakedleakedleaked1234567890"
    body = f"invalid key {key}; Authorization: Bearer {key}; api_key={key}"

    async with app_with(openai(lambda r: httpx.Response(401, text=body))) as client:
        chat = await client.post("/api/chat", json={"message": "hello"})
        stream = await client.post("/api/chat/stream", json={"message": "hello"})

    assert key not in chat.text and key not in stream.text


async def test_2d_credentials_inside_the_ollama_url_are_not_shown() -> None:
    provider = ollama(down, base_url="http://admin:hunter2@mock-ollama:11434")

    async with app_with(provider) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert "hunter2" not in res.text and "admin" not in res.text


# =================================================================== 3. odd model names in the request

ODD_MODELS = [
    "",
    "   ",
    "模型:latest",
    "x" * 5000,
    "a b",
    "m\nn",
    "`x`",
    "../../x",
    "m;rm -rf /",
    "$(id)",
]


@pytest.mark.parametrize("model", ODD_MODELS, ids=lambda m: repr(m)[:20])
async def test_3_any_model_string_gives_a_clean_error_and_never_a_command_injection(
    model: str,
) -> None:
    async with app_with(ollama(missing)) as client:
        res = await client.post("/api/chat", json={"message": "hello", "model": model})

    body = res.json()
    assert res.status_code == 502 and body["code"] == "model_not_found"
    hint = body["hint"]
    assert hint.count("`ollama pull ") == 1
    pulled = hint.split("`ollama pull ")[1].split("`")[0]
    assert pulled in ("<model>", "qwen3.5:4b", model.strip())
    assert not any(c in pulled for c in ";|$`\n ")


# ====================================================================== 4. memory switched on


async def test_4a_failed_turn_keeps_the_question_and_the_next_turn_recovers(memory) -> None:
    mode = {"ok": False}
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=OLLAMA_OK) if mode["ok"] else missing(request)

    async with app_with(ollama(handler), memory) as client:
        first = await client.post("/api/chat", json={"message": "what is rust?"})
        conv = (await memory.list_conversations()).conversations[0].id
        mode["ok"] = True
        second = await client.post(
            "/api/chat", json={"message": "and go?", "conversation_id": conv}
        )

    assert first.status_code == 502 and first.json()["code"] == "model_not_found"
    assert second.status_code == 200 and second.json()["reply"] == "Hello from the model"
    assert [m.role for m in await memory.get_messages(conv)] == ["user", "user", "assistant"]
    last_prompt = seen[-1]["messages"][-1]["content"]
    assert "what is rust?" in last_prompt and "and go?" in last_prompt


async def test_4b_a_failed_turn_never_saves_a_memory_but_the_retry_does(memory) -> None:
    mode = {"ok": False}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OLLAMA_OK) if mode["ok"] else down(request)

    async with app_with(ollama(handler), memory) as client:
        await client.post("/api/chat", json={"message": "Remember that I live in Pune."})
        assert (await memory.list_memories()).total == 0
        conv = (await memory.list_conversations()).conversations[0].id
        mode["ok"] = True
        await client.post(
            "/api/chat", json={"message": "Remember that I live in Pune.", "conversation_id": conv}
        )

    active = (await memory.list_memories(status="active")).memories
    assert [m.content for m in active] == ["I live in Pune."]


async def test_4c_failed_stream_with_memory_stores_no_assistant_message(memory) -> None:
    async with app_with(ollama(missing), memory) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    assert sse(res.text)[-1]["error"]["code"] == "model_not_found"
    conv = (await memory.list_conversations()).conversations[0].id
    assert [m.role for m in await memory.get_messages(conv)] == ["user"]


async def test_4d_reply_with_memory_keeps_latency_and_conversation_id(memory) -> None:
    async with app_with(ollama(lambda r: httpx.Response(200, json=OLLAMA_OK)), memory) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert body["conversation_id"] and isinstance(body["latency_ms"], int)


# ================================================================ 5. several requests at the same time


async def test_5a_ten_failing_chats_and_ten_failing_streams_at_once_all_get_a_full_payload() -> (
    None
):
    async with app_with(ollama(missing)) as client:
        chats = await asyncio.gather(
            *(client.post("/api/chat", json={"message": f"q{i}"}) for i in range(10))
        )
        streams = await asyncio.gather(
            *(client.post("/api/chat/stream", json={"message": f"s{i}"}) for i in range(10))
        )

    assert all(r.status_code == 502 and "ollama pull" in r.json()["hint"] for r in chats)
    assert all(sse(r.text)[-1]["error"]["code"] == "model_not_found" for r in streams)


async def test_5b_parallel_requests_each_report_their_own_latency() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.4 if b"slowone" in request.content else 0.03)
        return httpx.Response(200, json=OLLAMA_OK)

    async with app_with(ollama(handler)) as client:
        slow_res, fast_res = await asyncio.gather(
            client.post("/api/chat", json={"message": "slowone"}),
            client.post("/api/chat", json={"message": "quick"}),
        )

    assert slow_res.json()["latency_ms"] >= 350
    assert fast_res.json()["latency_ms"] < 300


async def test_5c_twenty_sequential_replies_all_have_a_sane_integer_latency() -> None:
    async with app_with(ollama(lambda r: httpx.Response(200, json=OLLAMA_OK))) as client:
        values = [
            (await client.post("/api/chat", json={"message": f"m{i}"})).json()["latency_ms"]
            for i in range(20)
        ]

    assert all(type(v) is int and 0 <= v < 5000 for v in values)


# ================================================================== 6. retries and fallback (real router)


async def test_6a_transient_errors_are_retried_and_the_latency_includes_every_attempt() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=OLLAMA_OK)

    async with app_with(make_router(ollama(handler), retries=2)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 200 and calls["n"] == 3
    assert res.json()["latency_ms"] >= 140


async def test_6b_when_retries_run_out_the_user_sees_a_clear_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="busy")

    async with app_with(make_router(ollama(handler), retries=2)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert calls["n"] == 3 and res.status_code == 502
    assert res.json()["code"] == "provider_response_error"


async def test_6c_local_down_cloud_fallback_answers_and_latency_covers_both_attempts() -> None:
    async def local_down(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        raise httpx.ConnectError("refused")

    async def cloud_ok(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=OPENAI_OK)

    router = make_router(ollama(local_down), openai(cloud_ok))
    async with app_with(router) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == 200 and body["provider"] == "openai"
    assert body["reply"] == "Hello from the cloud" and body["latency_ms"] >= 180


async def test_6d_stream_falls_back_too_and_reports_latency_once() -> None:
    def cloud_stream(request: httpx.Request) -> httpx.Response:
        events = [
            {
                "model": "gpt-4o-mini",
                "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}],
            },
            {"model": "gpt-4o-mini", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body)

    router = make_router(ollama(down), openai(cloud_stream))
    async with app_with(router) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = sse(res.text)
    providers = {e["provider"] for e in events if e.get("provider")}
    assert providers == {"openai"}
    assert sum(e["latency_ms"] is not None for e in events) == 1 and "[DONE]" in res.text


async def test_6e_a_missing_local_model_is_permanent_so_the_cloud_is_not_used() -> None:
    cloud_calls = {"n": 0}

    def cloud(request: httpx.Request) -> httpx.Response:
        cloud_calls["n"] += 1
        return httpx.Response(200, json=OPENAI_OK)

    async with app_with(make_router(ollama(missing), openai(cloud))) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert cloud_calls["n"] == 0, "no silent switch to the cloud for a fixable local problem"
    assert res.json()["code"] == "model_not_found"
    assert "ollama pull qwen3.5:4b" in res.json()["hint"]


async def test_6f_if_the_cloud_fallback_fails_too_the_user_sees_the_last_error() -> None:
    router = make_router(ollama(down), openai(status(401, "bad key")))
    async with app_with(router) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert body["provider"] == "openai" and body["code"] == "authentication_failed"
    assert "OPENAI_API_KEY" in body["hint"]


# ===================================================================== 7. other endpoints and auth


async def test_7a_listing_models_while_ollama_is_down_gets_the_hint_and_health_stays_200() -> None:
    async with app_with(ollama(down)) as client:
        models = await client.get("/api/models")
        health = await client.get("/health")

    assert models.status_code == 503 and "ollama serve" in models.json()["hint"]
    assert health.status_code == 200 and health.json()["status"] == "degraded"


async def test_7b_a_server_that_is_not_ollama_is_not_told_to_pull_a_model() -> None:
    """GET /api/tags answering 404 means a wrong OLLAMA_BASE_URL, not a missing model."""
    async with app_with(ollama(lambda r: httpx.Response(404, text="404 page not found"))) as client:
        res = await client.get("/api/models")

    body = res.json()
    assert res.status_code == 502
    assert body["code"] != "model_not_found" and "ollama pull" not in body.get("hint", "")


async def test_7c_with_an_api_token_provider_details_need_the_token() -> None:
    async with app_with(ollama(missing), api_token=TOKEN) as client:
        anonymous = await client.post("/api/chat", json={"message": "hello"})
        authorised = await client.post(
            "/api/chat", json={"message": "hello"}, headers={"authorization": f"Bearer {TOKEN}"}
        )

    assert anonymous.status_code == 401
    assert "hint" not in anonymous.text and "ollama" not in anonymous.text
    assert authorised.status_code == 502 and authorised.json()["code"] == "model_not_found"


async def test_7d_a_working_model_after_a_failing_one_just_works() -> None:
    mode = {"ok": False}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OLLAMA_OK) if mode["ok"] else down(request)

    async with app_with(ollama(handler)) as client:
        before = await client.post("/api/chat", json={"message": "hello"})
        mode["ok"] = True
        after = await client.post("/api/chat", json={"message": "hello"})

    assert before.status_code == 503 and after.status_code == 200 and "hint" not in after.json()


# ================================================================= 8. latency in the contract itself


async def test_8a_openapi_documents_latency_ms_on_the_reply() -> None:
    async with app_with(ollama(down)) as client:
        schemas = (await client.get("/openapi.json")).json()["components"]["schemas"]

    prop = json.dumps(schemas["BrainChatResponse"]["properties"]["latency_ms"])
    assert "integer" in prop and "milliseconds" in prop.lower()
    # the SSE chunk is not a response_model, so it is documented in the LLD instead
    assert "BrainStreamChunk" not in schemas


async def test_8b_a_two_hundred_chunk_stream_has_exactly_one_latency_on_the_last_chunk() -> None:
    class Long(_NoExtraction):
        async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
            for i in range(200):
                last = i == 199
                yield ChatStreamChunk(
                    delta=f"w{i} ", model="m", provider="p", finish_reason="stop" if last else None
                )

    async with app_with(Long()) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = sse(res.text)
    assert len(events) == 200 and events[-1]["latency_ms"] is not None
    assert all(e["latency_ms"] is None for e in events[:-1])


async def test_8c_slow_memory_lookup_and_extraction_do_not_inflate_the_model_latency(
    memory, monkeypatch
) -> None:
    async def slow_lookup(*args, **kwargs):
        await asyncio.sleep(0.25)
        return []

    monkeypatch.setattr(memory, "get_context_memories", slow_lookup)

    class SlowExtraction(_NoExtraction):
        async def generate(self, request: ChatRequest) -> ChatResponse:
            await asyncio.sleep(0.25)
            return await super().generate(request)

    memory.extractor = InferredMemoryExtractor(settings=memory.settings, provider=SlowExtraction())

    async with app_with(ollama(lambda r: httpx.Response(200, json=OLLAMA_OK)), memory) as client:
        started = asyncio.get_running_loop().time()
        res = await client.post("/api/chat", json={"message": "I really like green tea these days"})
        wall = asyncio.get_running_loop().time() - started

    assert wall >= 0.45, "the memory work really was slow"
    assert res.json()["latency_ms"] < 200, "latency measures only the model call"


# ============================================================ 9. a wrong address is not a missing model


async def test_9a_chat_against_a_server_that_is_not_ollama_points_at_the_address() -> None:
    async with app_with(ollama(lambda r: httpx.Response(404, text="404 page not found"))) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert body["code"] == "endpoint_not_found" and "model" not in body
    assert "OLLAMA_BASE_URL" in body["hint"] and "ollama pull" not in body["hint"]


async def test_9b_the_same_for_a_stream() -> None:
    async with app_with(ollama(lambda r: httpx.Response(404, text="Not Found"))) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    error = sse(res.text)[-1]["error"]
    assert error["code"] == "endpoint_not_found" and "ollama pull" not in error["hint"]


async def test_9c_ollama_naming_the_model_in_its_404_is_still_a_missing_model() -> None:
    body = json.dumps({"error": "model 'llama9' not found, try pulling it first"})
    async with app_with(ollama(lambda r: httpx.Response(404, text=body))) as client:
        res = await client.post("/api/chat", json={"message": "hello", "model": "llama9"})

    assert res.json()["code"] == "model_not_found" and "ollama pull llama9" in res.json()["hint"]


async def test_9d_a_cloud_model_listing_404_points_at_the_base_url_not_the_model() -> None:
    async with app_with(openai(lambda r: httpx.Response(404, text="nope"))) as client:
        res = await client.get("/api/models")

    body = res.json()
    assert body["code"] == "endpoint_not_found" and "OPENAI_BASE_URL" in body["hint"]


async def test_9e_a_cloud_chat_404_is_still_a_model_problem() -> None:
    async with app_with(openai(lambda r: httpx.Response(404, text="nope"))) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert res.json()["code"] == "model_not_found" and "OPENAI_MODEL" in res.json()["hint"]
