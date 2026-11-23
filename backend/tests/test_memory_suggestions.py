"""Decision D-001: inferred memories are suggestions that become active only on user approval."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.api.deps import get_memory_service
from app.core.config import Settings
from app.main import app
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.exceptions import MemoryNotFoundError, MemoryValidationError
from app.memory.extractor import InferredMemoryExtractor
from app.memory.schemas import MemoryApprove, MemoryCreate, MemoryUpdate
from app.memory.service import MemoryService
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)


class ScriptedProvider(LLMProvider):
    """Returns a fixed JSON extraction payload."""

    def __init__(self, items: list[dict]) -> None:
        self.payload = json.dumps(items)

    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content=self.payload, model="m", provider="ollama")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta=self.payload, model="m", provider="ollama")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="ollama")

    async def aclose(self) -> None:
        pass


def _item(content: str, confidence: float = 0.95) -> dict:
    return {
        "content": content,
        "category": "preference",
        "confidence": confidence,
        "importance": 3.0,
    }


@pytest_asyncio.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker]:
    engine: AsyncEngine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 's.db'}")
    await init_db(engine)
    try:
        yield get_session_factory(engine)
    finally:
        await engine.dispose()


def _service(factory: async_sessionmaker, items: list[dict], **settings) -> MemoryService:
    cfg = Settings(_env_file=None, **settings)
    svc = MemoryService(session_factory=factory, settings=cfg)
    svc.extractor = InferredMemoryExtractor(settings=cfg, provider=ScriptedProvider(items))
    return svc


async def _suggest(svc: MemoryService, text: str = "I like tea a lot these days."):
    stored = await svc.extract_and_store_memories(text)
    assert len(stored) == 1
    return stored[0]


# ---------------------------------------------------------------- suggestions are never active


async def test_high_confidence_inference_is_still_only_a_suggestion(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.", confidence=1.0)])

    mem = await _suggest(svc)

    assert mem.origin == "inferred" and mem.status == "candidate"
    assert (await svc.list_memories(status="active")).total == 0


async def test_explicit_directive_is_still_active_immediately(factory) -> None:
    svc = _service(factory, [])

    stored = await svc.extract_and_store_memories("Remember that I prefer tabs over spaces.")

    assert len(stored) == 1
    assert stored[0].origin == "explicit" and stored[0].status == "active"


async def test_suggestions_disabled_creates_no_inferred_memories(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")], memory_suggestions_enabled=False)

    assert await svc.extract_and_store_memories("I like tea a lot these days.") == []
    assert (await svc.list_memories()).total == 0


async def test_suggestions_disabled_still_allows_explicit_memory(factory) -> None:
    svc = _service(factory, [], memory_suggestions_enabled=False)

    stored = await svc.extract_and_store_memories("Remember that I prefer tabs over spaces.")

    assert len(stored) == 1 and stored[0].status == "active"


async def test_identical_pending_suggestion_is_not_stacked(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])

    first = await _suggest(svc)
    again = await svc.extract_and_store_memories("I like tea a lot these days.")

    assert again == []
    assert (await svc.list_memories(status="candidate")).total == 1
    assert first.status == "candidate"


# ---------------------------------------------------------------- approve


async def test_approve_makes_suggestion_active(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await _suggest(svc)

    approved = await svc.approve_memory(mem.id)

    assert approved.status == "active" and approved.origin == "inferred"
    assert approved.metadata_json["approved_by_user"] is True
    assert [m.id for m in (await svc.list_memories(status="active")).memories] == [mem.id]


async def test_approve_can_edit_content(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await _suggest(svc)

    approved = await svc.approve_memory(mem.id, MemoryApprove(content="User likes jasmine tea."))

    assert approved.content == "User likes jasmine tea." and approved.status == "active"


async def test_approve_edit_still_blocks_sensitive_content() -> None:
    with pytest.raises(ValueError):
        MemoryApprove(content="my password is hunter2")


async def test_approve_only_works_on_pending_suggestions(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await _suggest(svc)
    await svc.approve_memory(mem.id)

    with pytest.raises(MemoryValidationError, match="not a pending suggestion"):
        await svc.approve_memory(mem.id)
    with pytest.raises(MemoryNotFoundError):
        await svc.approve_memory("does-not-exist")


async def test_update_cannot_bypass_approval(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await _suggest(svc)

    with pytest.raises(MemoryValidationError, match="approve the suggestion"):
        await svc.update_memory(mem.id, MemoryUpdate(status="active"))
    with pytest.raises(MemoryValidationError, match="reject action"):
        await svc.update_memory(mem.id, MemoryUpdate(status="rejected"))
    assert (await svc.get_memory(mem.id)).status == "candidate"  # type: ignore[union-attr]


async def test_approved_memory_can_still_be_edited(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await svc.approve_memory((await _suggest(svc)).id)

    edited = await svc.update_memory(mem.id, MemoryUpdate(importance=5.0))

    assert edited.status == "active" and edited.importance == 5.0


# ---------------------------------------------------------------- conflicts


async def test_conflicting_suggestion_waits_and_approval_supersedes_the_old_memory(factory) -> None:
    svc = _service(factory, [_item("User prefers light mode UI.")])
    old = await svc.create_memory(
        MemoryCreate(content="User prefers dark mode UI.", category="preference")
    )

    suggestion = await _suggest(svc, "Light mode is easier on my eyes lately.")
    assert suggestion.status == "candidate"
    assert suggestion.metadata_json["conflicts_with"] == old.id
    assert (await svc.get_memory(old.id)).status == "active"  # type: ignore[union-attr]

    await svc.approve_memory(suggestion.id)

    old_now = await svc.get_memory(old.id)
    assert old_now is not None
    assert old_now.status == "superseded" and old_now.superseded_by_id == suggestion.id
    assert [m.id for m in (await svc.list_memories(status="active")).memories] == [suggestion.id]


# ---------------------------------------------------------------- reject


async def test_reject_marks_suggestion_and_hides_it_from_active(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    mem = await _suggest(svc)

    rejected = await svc.reject_memory(mem.id)

    assert rejected.status == "rejected"
    assert rejected.metadata_json["rejected_by_user"] is True
    assert (await svc.list_memories(status="active")).total == 0
    assert (await svc.list_memories(status="candidate")).total == 0


async def test_rejected_suggestion_is_never_suggested_again(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    await svc.reject_memory((await _suggest(svc)).id)

    assert await svc.extract_and_store_memories("I like tea a lot these days.") == []
    assert (await svc.list_memories(status="candidate")).total == 0
    assert (await svc.list_memories(status="rejected")).total == 1


async def test_explicit_request_overrides_an_earlier_rejection(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    await svc.reject_memory((await _suggest(svc)).id)

    stored = await svc.extract_and_store_memories("Remember that User likes green tea.")

    assert len(stored) == 1 and stored[0].origin == "explicit" and stored[0].status == "active"


async def test_reject_only_works_on_pending_suggestions(factory) -> None:
    svc = _service(factory, [])
    active = await svc.create_memory(MemoryCreate(content="User uses Linux."))

    with pytest.raises(MemoryValidationError, match="not a pending suggestion"):
        await svc.reject_memory(active.id)
    with pytest.raises(MemoryNotFoundError):
        await svc.reject_memory("does-not-exist")


# ---------------------------------------------------------------- HTTP API


async def test_approve_and_reject_endpoints(factory) -> None:
    svc = _service(factory, [_item("User likes green tea.")])
    to_approve = await _suggest(svc)
    svc.extractor = InferredMemoryExtractor(
        settings=svc.settings, provider=ScriptedProvider([_item("User dislikes coffee.")])
    )
    to_reject = await _suggest(svc, "I really dislike coffee, honestly.")

    app.dependency_overrides[get_memory_service] = lambda: svc
    app.state.memory_service = svc
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            pending = await client.get("/api/memories", params={"status": "candidate"})
            assert pending.status_code == 200 and pending.json()["total"] == 2

            res = await client.post(
                f"/api/memories/{to_approve.id}/approve", json={"content": "User likes oolong tea."}
            )
            assert res.status_code == 200, res.text
            assert res.json()["status"] == "active"
            assert res.json()["content"] == "User likes oolong tea."

            res = await client.post(f"/api/memories/{to_reject.id}/reject")
            assert res.status_code == 200 and res.json()["status"] == "rejected"

            # Approve without a body works too; a non-suggestion is a 400, unknown id a 404.
            assert (await client.post(f"/api/memories/{to_approve.id}/approve")).status_code == 400
            assert (await client.post("/api/memories/nope/approve")).status_code == 404
            assert (await client.post("/api/memories/nope/reject")).status_code == 404

            bad = await client.patch(f"/api/memories/{to_reject.id}", json={"status": "active"})
            assert bad.status_code == 400
            sensitive = await client.post(
                f"/api/memories/{to_approve.id}/approve", json={"content": "my password is abc123"}
            )
            assert sensitive.status_code == 422
    finally:
        app.dependency_overrides.clear()
        app.state.memory_service = None
