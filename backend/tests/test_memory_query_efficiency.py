"""Regression tests for review finding #10 (performance part).

`Conversation.messages` used `lazy="selectin"`, so any query returning a conversation also loaded
its entire message history. These tests assert on what is loaded (SQL statements and Message
entities), which is deterministic, instead of on wall-clock time, which is not.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, func, select
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.memory.database import create_database_engine, get_session_factory, init_db
from app.memory.models import Base, Conversation, Memory, Message
from app.memory.repository import ConversationRepository, MemoryRepository
from app.memory.service import MemoryService

N_MESSAGES = 120


@dataclass
class Probe:
    statements: list[str] = field(default_factory=list)
    messages_loaded: int = 0

    def reset(self) -> None:
        self.statements.clear()
        self.messages_loaded = 0

    @property
    def message_row_selects(self) -> list[str]:
        """SELECTs that fetch message ROWS (a COUNT over messages is fine and excluded)."""
        return [
            s
            for s in self.statements
            if s.lstrip().upper().startswith("SELECT")
            and "FROM messages" in s
            and "count(" not in s.lower()
        ]


@pytest_asyncio.fixture
async def seeded(
    tmp_path: Path,
) -> AsyncIterator[tuple[MemoryService, async_sessionmaker, Probe, str]]:
    engine: AsyncEngine = create_database_engine(f"sqlite+aiosqlite:///{tmp_path / 'q.db'}")
    await init_db(engine)
    factory = get_session_factory(engine)
    service = MemoryService(session_factory=factory, max_history_messages=20)

    async with factory() as session, session.begin():
        crepo = ConversationRepository(session)
        big = await crepo.create_conversation(title="big")
        for i in range(N_MESSAGES):
            await crepo.save_message(big.id, "user" if i % 2 == 0 else "assistant", f"m{i}")
        for k in range(5):
            c = await crepo.create_conversation(title=f"small {k}")
            await crepo.save_message(c.id, "user", "hi")
        mrepo = MemoryRepository(session)
        for k in range(5):
            await mrepo.create_memory(
                content=f"fact {k}",
                category="fact",
                origin="inferred",
                confidence=0.9,
                importance=3.0,
                status="active",
                source_conversation_id=big.id,
            )
        big_id = big.id

    probe = Probe()

    def on_execute(conn, cursor, statement, params, context, executemany):
        probe.statements.append(statement)

    def on_load(target, context):
        probe.messages_loaded += 1

    event.listen(engine.sync_engine, "before_cursor_execute", on_execute)
    event.listen(Message, "load", on_load)
    try:
        yield service, factory, probe, big_id
    finally:
        event.remove(Message, "load", on_load)
        await engine.dispose()


# ----------------------------------------------------------- nothing loads history implicitly


async def test_save_message_does_not_load_history(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    await service.save_message(big_id, "assistant", "ok")
    assert probe.messages_loaded == 0
    assert probe.message_row_selects == []


async def test_get_conversation_summary_does_not_load_history(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    info = await service.get_conversation(big_id, include_messages=False)
    assert probe.messages_loaded == 0
    assert info.message_count == N_MESSAGES


async def test_update_conversation_does_not_load_history(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    info = await service.update_conversation(big_id, title="renamed")
    assert probe.messages_loaded == 0
    assert info is not None and info.title == "renamed" and info.message_count == N_MESSAGES


async def test_list_conversations_does_not_load_history(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    page = await service.list_conversations(limit=50)
    assert probe.messages_loaded == 0
    assert probe.message_row_selects == []
    counts = {c.id: c.message_count for c in page.conversations}
    assert counts[big_id] == N_MESSAGES
    assert sorted(counts.values()) == [1, 1, 1, 1, 1, N_MESSAGES]


async def test_list_memories_does_not_load_source_conversation_history(seeded) -> None:
    service, _, probe, _ = seeded
    probe.reset()
    listing = await service.list_memories(limit=50)
    assert listing.total == 5
    assert probe.messages_loaded == 0
    assert probe.message_row_selects == []


# --------------------------------------------- bounded reads load only what they were asked for


async def test_recent_history_loads_only_the_requested_window(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    history = await service.get_recent_history(big_id, limit=20)
    assert len(history) == 20
    assert probe.messages_loaded == 20
    assert history[-1].content == f"m{N_MESSAGES - 1}"  # newest last, chronological


async def test_get_messages_loads_only_the_requested_slice(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    msgs = await service.get_messages(big_id, limit=10, offset=5)
    assert [m.content for m in msgs] == [f"m{i}" for i in range(5, 15)]
    assert probe.messages_loaded == 10


async def test_prepare_turn_loads_only_the_history_window(seeded) -> None:
    service, _, probe, big_id = seeded
    probe.reset()
    conv_id, history = await service.prepare_turn(big_id, "next question")
    assert conv_id == big_id
    assert len(history) == 20
    assert probe.messages_loaded == 20


# ------------------------------------------------------- behavior that must NOT change


async def test_explicit_full_thread_load_still_returns_everything(seeded) -> None:
    service, _, _, big_id = seeded
    detail = await service.get_conversation(big_id, include_messages=True)
    assert detail is not None
    assert len(detail.messages) == N_MESSAGES == detail.message_count
    assert [m.content for m in detail.messages[:3]] == ["m0", "m1", "m2"]


async def test_unknown_conversation_behaviour_unchanged(seeded) -> None:
    service, _, _, _ = seeded
    assert await service.get_conversation("nope", include_messages=False) is None
    assert await service.update_conversation("nope", title="x") is None
    assert await service.delete_conversation("nope") is False


async def test_delete_conversation_removes_all_messages_and_nulls_memory_source(seeded) -> None:
    service, factory, probe, big_id = seeded
    probe.reset()
    assert await service.delete_conversation(big_id) is True
    assert probe.messages_loaded == 0  # deleted without loading the thread
    async with factory() as session:
        left = (
            await session.execute(
                select(func.count(Message.id)).where(Message.conversation_id == big_id)
            )
        ).scalar()
        sources = (await session.execute(select(Memory.source_conversation_id))).scalars().all()
    assert left == 0
    assert sources and all(s is None for s in sources)  # FK SET NULL preserved


async def test_delete_conversation_does_not_depend_on_fk_pragma(tmp_path: Path) -> None:
    """A plain engine WITHOUT the foreign_keys pragma must still not leave orphan messages."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'nofk.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = MemoryService(session_factory=factory)
    try:
        conv = await service.create_conversation(title="t")
        for _ in range(5):
            await service.save_message(conv.id, "user", "hello")
        assert await service.delete_conversation(conv.id) is True
        async with factory() as session:
            left = (await session.execute(select(func.count(Message.id)))).scalar()
        assert left == 0
    finally:
        await engine.dispose()


async def test_repository_count_messages(seeded) -> None:
    _, factory, _, big_id = seeded
    async with factory() as session:
        repo = ConversationRepository(session)
        assert await repo.count_messages(big_id) == N_MESSAGES
        assert await repo.count_messages("missing") == 0


# ---------------------------------------------- guard: implicit loading now fails loudly


async def test_relationships_are_never_lazily_loaded_by_accident(seeded) -> None:
    _, factory, _, big_id = seeded
    async with factory() as session:
        conv = (
            await session.execute(select(Conversation).where(Conversation.id == big_id))
        ).scalar_one()
        with pytest.raises(InvalidRequestError):
            _ = conv.messages

        mem = (await session.execute(select(Memory).limit(1))).scalar_one()
        with pytest.raises(InvalidRequestError):
            _ = mem.source_conversation
        with pytest.raises(InvalidRequestError):
            _ = mem.superseded_by
