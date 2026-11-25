"""Superseded-memory lifecycle when a replacement is deleted (final audit finding 3).

Invariant under test:
  1. A hard delete never changes a predecessor's status: nothing is ever reactivated.
  2. If the purged memory had a live successor, predecessors are re-linked to it
     (A -> B -> C, purge B  =>  A -> C).
  3. Otherwise predecessors carry an explicit tombstone in metadata.
  4. "Cannot enter superseded without a pointer" still holds for transitions INTO the state,
     but an already-superseded memory stays editable and its pointer can never be cleared.
  5. Soft delete is unchanged (pointer kept).
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.api.deps import get_memory_service
from app.main import app
from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.exceptions import MemoryValidationError
from app.memory.extractor import InferredMemoryExtractor
from app.memory.models import Memory
from app.memory.repository import FORMER_SUCCESSOR_KEY, SUPERSEDED_BY_DELETED_KEY
from app.memory.schemas import MemoryCreate, MemoryRead, MemoryUpdate
from app.memory.service import MemoryService
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderHealth,
)


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[tuple[MemoryService, async_sessionmaker]]:
    engine: AsyncEngine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'life.db'}")
    await init_db(engine)
    factory = get_session_factory(engine)
    try:
        yield MemoryService(session_factory=factory), factory
    finally:
        await engine.dispose()


async def make(svc: MemoryService, text: str, origin: str = "explicit") -> MemoryRead:
    return await svc.create_memory(
        MemoryCreate(
            content=text,
            category="fact",
            origin=origin,  # type: ignore[arg-type]
            confidence=1.0 if origin == "explicit" else 0.9,
            importance=3.0,
            status="active",
        )
    )


async def supersede(svc: MemoryService, old: MemoryRead, new: MemoryRead) -> None:
    await svc.update_memory(old.id, MemoryUpdate(status="superseded", superseded_by_id=new.id))


async def get(svc: MemoryService, mem: MemoryRead) -> MemoryRead:
    current = await svc.get_memory(mem.id)
    assert current is not None
    return current


async def active_ids(svc: MemoryService) -> set[str]:
    return {m.id for m in (await svc.list_memories(status="active")).memories}


def assert_tombstoned(mem: MemoryRead, former: MemoryRead) -> None:
    assert mem.status == "superseded"
    assert mem.superseded_by_id is None
    assert mem.metadata_json[SUPERSEDED_BY_DELETED_KEY] is True
    assert mem.metadata_json[FORMER_SUCCESSOR_KEY] == former.id


# ============================================================================ basic case


async def test_hard_deleting_the_replacement_keeps_the_old_memory_superseded(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)

    result = await svc.delete_memory(b.id, hard=True)

    assert result.hard is True
    assert await svc.get_memory(b.id) is None
    assert_tombstoned(await get(svc, a), b)
    assert a.id not in await active_ids(svc)  # NOT reactivated


async def test_orphaned_memory_stays_editable_and_superseded(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.delete_memory(b.id, hard=True)

    updated = await svc.update_memory(a.id, MemoryUpdate(importance=4.5))  # was: rejected

    assert updated.importance == 4.5
    assert updated.status == "superseded"
    assert updated.metadata_json[FORMER_SUCCESSOR_KEY] == b.id  # tombstone survives an edit
    assert a.id not in await active_ids(svc)


async def test_replacement_listing_and_other_memories_are_unaffected(env) -> None:
    svc, _ = env
    a, b, other = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User likes tea."),
    )
    await supersede(svc, a, b)
    await svc.delete_memory(b.id, hard=True)
    assert await active_ids(svc) == {other.id}
    untouched = await get(svc, other)
    assert untouched.status == "active" and not untouched.metadata_json


# ================================================================================ chains


async def test_chain_middle_deleted_relinks_predecessor_to_live_successor(env) -> None:
    svc, _ = env
    a, b, c = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User uses Helix."),
    )
    await supersede(svc, a, b)
    await supersede(svc, b, c)

    await svc.delete_memory(b.id, hard=True)

    a_now = await get(svc, a)
    assert a_now.status == "superseded"
    assert a_now.superseded_by_id == c.id  # lineage now points at the live head
    assert SUPERSEDED_BY_DELETED_KEY not in a_now.metadata_json  # no tombstone needed
    assert (await get(svc, c)).status == "active"
    assert await active_ids(svc) == {c.id}


async def test_chain_head_deleted_tombstones_only_its_direct_predecessor(env) -> None:
    svc, _ = env
    a, b, c = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User uses Helix."),
    )
    await supersede(svc, a, b)
    await supersede(svc, b, c)

    await svc.delete_memory(c.id, hard=True)

    b_now, a_now = await get(svc, b), await get(svc, a)
    assert_tombstoned(b_now, c)
    assert a_now.status == "superseded" and a_now.superseded_by_id == b.id  # A -> B kept
    assert await active_ids(svc) == set()  # nothing resurrected by the purge


async def test_long_chain_compresses_step_by_step_and_ends_in_a_tombstone(env) -> None:
    svc, _ = env
    a, b, c, d = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User uses Helix."),
        await make(svc, "User uses Zed."),
    )
    await supersede(svc, a, b)
    await supersede(svc, b, c)
    await supersede(svc, c, d)

    await svc.delete_memory(b.id, hard=True)
    assert (await get(svc, a)).superseded_by_id == c.id
    await svc.delete_memory(c.id, hard=True)
    assert (await get(svc, a)).superseded_by_id == d.id
    await svc.delete_memory(d.id, hard=True)

    assert_tombstoned(await get(svc, a), d)
    assert await active_ids(svc) == set()


async def test_deleting_the_oldest_link_relinks_what_pointed_at_it(env) -> None:
    svc, _ = env
    z, a, b = (
        await make(svc, "User uses Emacs."),
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
    )
    await supersede(svc, z, a)
    await supersede(svc, a, b)

    await svc.delete_memory(a.id, hard=True)

    assert (await get(svc, z)).superseded_by_id == b.id
    assert (await get(svc, b)).status == "active"


async def test_multiple_predecessors_are_all_handled(env) -> None:
    svc, _ = env
    a1, a2, b = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Emacs."),
        await make(svc, "User uses Neovim."),
    )
    await supersede(svc, a1, b)
    await supersede(svc, a2, b)

    await svc.delete_memory(b.id, hard=True)

    assert_tombstoned(await get(svc, a1), b)
    assert_tombstoned(await get(svc, a2), b)
    assert await active_ids(svc) == set()


# ================================================================== soft delete (unchanged)


async def test_soft_delete_keeps_the_pointer_and_nothing_is_reactivated(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)

    await svc.delete_memory(b.id, hard=False)

    a_now = await get(svc, a)
    assert a_now.status == "superseded" and a_now.superseded_by_id == b.id
    assert SUPERSEDED_BY_DELETED_KEY not in a_now.metadata_json
    assert (await get(svc, b)).status == "deleted"
    assert await active_ids(svc) == set()


async def test_hard_deleting_a_soft_deleted_replacement_still_tombstones(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.delete_memory(b.id, hard=False)
    await svc.delete_memory(b.id, hard=True)
    assert_tombstoned(await get(svc, a), b)


async def test_hard_deleting_an_unrelated_or_leaf_memory_changes_nothing_else(env) -> None:
    svc, _ = env
    a, b, lone = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User likes tea."),
    )
    await supersede(svc, a, b)
    await svc.delete_memory(lone.id, hard=True)
    a_now = await get(svc, a)
    assert a_now.superseded_by_id == b.id and not a_now.metadata_json


async def test_deleting_the_superseded_memory_itself_leaves_the_replacement_alone(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.delete_memory(a.id, hard=True)
    assert (await get(svc, b)).status == "active"


# ============================================================ never silently reactivated


class _NoModel(LLMProvider):
    async def generate(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(content="[]", model="m", provider="ollama")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta="[]")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="ollama")

    async def aclose(self) -> None:
        pass


async def test_restating_a_superseded_fact_creates_a_new_memory_not_a_revival(env) -> None:
    svc, _ = env
    svc.extractor = InferredMemoryExtractor(provider=_NoModel())
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.delete_memory(b.id, hard=True)

    stored = await svc.extract_and_store_memories("Remember that User uses Vim.")

    assert stored[0].id != a.id and stored[0].status == "active"
    a_now = await get(svc, a)
    assert a_now.status == "superseded"  # the old row is not revived or deduplicated into
    assert await active_ids(svc) == {stored[0].id}


async def test_only_an_explicit_status_update_can_reactivate(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.delete_memory(b.id, hard=True)
    assert a.id not in await active_ids(svc)

    revived = await svc.update_memory(a.id, MemoryUpdate(status="active"))  # deliberate action

    assert revived.status == "active"


# ============================================================== validation still enforced


def test_cannot_set_superseded_without_pointer_via_schema() -> None:
    with pytest.raises(ValidationError):
        MemoryUpdate(status="superseded")


async def test_active_memory_cannot_enter_superseded_without_a_valid_pointer(env) -> None:
    svc, _ = env
    a = await make(svc, "User uses Vim.")
    with pytest.raises(MemoryValidationError):
        await svc.update_memory(a.id, MemoryUpdate(status="superseded", superseded_by_id="nope"))
    with pytest.raises(MemoryValidationError):
        await svc.update_memory(a.id, MemoryUpdate(superseded_by_id=a.id))  # self-reference
    assert (await get(svc, a)).status == "active"


async def test_service_guard_blocks_entering_superseded_even_if_schema_is_bypassed(env) -> None:
    """Defense in depth: the service re-checks the rule that MemoryUpdate's validator enforces."""
    svc, _ = env
    a = await make(svc, "User uses Vim.")
    bypassed = MemoryUpdate.model_construct(status="superseded", superseded_by_id=None)

    with pytest.raises(MemoryValidationError, match="superseded_by_id"):
        await svc.update_memory(a.id, bypassed)

    assert (await get(svc, a)).status == "active"


async def test_pointer_cannot_be_cleared_through_the_api(env) -> None:
    svc, _ = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    await svc.update_memory(a.id, MemoryUpdate(importance=2.0))
    assert (await get(svc, a)).superseded_by_id == b.id  # editing never drops the pointer


async def test_legacy_orphan_without_tombstone_is_editable_but_not_reactivated(env) -> None:
    """Rows orphaned by the old FK-only behaviour (superseded, NULL pointer, no marker)."""
    svc, factory = env
    a, b = await make(svc, "User uses Vim."), await make(svc, "User uses Neovim.")
    await supersede(svc, a, b)
    async with factory() as session, session.begin():
        await session.execute(update(Memory).where(Memory.id == a.id).values(superseded_by_id=None))

    updated = await svc.update_memory(a.id, MemoryUpdate(importance=1.5))

    assert updated.status == "superseded" and updated.importance == 1.5
    assert a.id not in await active_ids(svc)


# ============================================================================ HTTP endpoint


async def test_hard_delete_endpoint_relinks_and_tombstones(env) -> None:
    svc, _ = env
    a, b, c = (
        await make(svc, "User uses Vim."),
        await make(svc, "User uses Neovim."),
        await make(svc, "User uses Helix."),
    )
    await supersede(svc, a, b)
    await supersede(svc, b, c)

    app.dependency_overrides[get_memory_service] = lambda: svc
    app.state.memory_service = svc
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://localhost"
        ) as client:
            res = await client.delete(f"/api/memories/{b.id}", params={"hard": "true"})
            assert res.status_code == 200 and res.json()["hard"] is True
            relinked = (await client.get(f"/api/memories/{a.id}")).json()
            assert relinked["status"] == "superseded"
            assert relinked["superseded_by_id"] == c.id

            res = await client.delete(f"/api/memories/{c.id}", params={"hard": "true"})
            assert res.status_code == 200
            tomb = (await client.get(f"/api/memories/{a.id}")).json()
            assert tomb["status"] == "superseded" and tomb["superseded_by_id"] is None
            assert tomb["metadata_json"][SUPERSEDED_BY_DELETED_KEY] is True
            assert tomb["metadata_json"][FORMER_SUCCESSOR_KEY] == c.id

            edit = await client.patch(f"/api/memories/{a.id}", json={"importance": 4.0})
            assert edit.status_code == 200, edit.text
            assert json.loads(edit.text)["status"] == "superseded"
    finally:
        app.dependency_overrides.clear()
        app.state.memory_service = None
