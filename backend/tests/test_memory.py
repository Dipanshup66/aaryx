"""Unit and integration tests for persistent conversation memory."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.memory.database import (
    create_database_engine,
    ensure_db_directory,
    get_session_factory,
    init_db,
    is_file_based_sqlite,
)
from app.memory.exceptions import ConversationNotFoundError, ConversationValidationError
from app.memory.repository import ConversationRepository
from app.memory.schemas import ConversationDetail
from app.memory.service import MemoryService


@pytest_asyncio.fixture
async def temp_db(
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncEngine, async_sessionmaker[AsyncSession]]]:
    """Provide a fresh temporary SQLite database engine and sessionmaker."""
    db_file = tmp_path / "test_aaryx.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_database_engine(db_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)

    try:
        yield engine, session_factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_database_initialization(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify tables exist and SQLite foreign keys are active."""
    _engine, session_factory = temp_db

    async with session_factory() as session:
        # Check foreign keys pragma
        fk_res = await session.execute(text("PRAGMA foreign_keys;"))
        assert fk_res.scalar() == 1

        # Check tables exist
        tbl_res = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        )
        table_names = [row[0] for row in tbl_res.fetchall()]
        assert "conversations" in table_names
        assert "messages" in table_names


@pytest.mark.asyncio
async def test_conversation_creation(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify conversation creation with default and custom attributes."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    conv = await service.create_conversation(
        title="Project Planning",
        metadata={"tags": ["architecture", "ai"]},
    )

    assert conv.id is not None
    assert conv.title == "Project Planning"
    assert conv.metadata_json == {"tags": ["architecture", "ai"]}
    assert conv.created_at is not None
    assert conv.updated_at is not None
    assert conv.message_count == 0


@pytest.mark.asyncio
async def test_conversation_retrieval(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify retrieving a conversation summary and full conversation details."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    conv = await service.create_conversation(title="Test Retrieval")
    await service.save_message(
        conversation_id=conv.id,
        role="user",
        content="What is AARYX?",
    )
    await service.save_message(
        conversation_id=conv.id,
        role="assistant",
        content="AARYX is your adaptive AI companion.",
    )

    # Summary retrieval
    summary = await service.get_conversation(conv.id, include_messages=False)
    assert summary is not None
    assert summary.id == conv.id
    assert summary.message_count == 2

    # Detailed retrieval with messages
    detail = await service.get_conversation(conv.id, include_messages=True)
    assert isinstance(detail, ConversationDetail)
    assert len(detail.messages) == 2
    assert detail.messages[0].role == "user"
    assert detail.messages[0].content == "What is AARYX?"
    assert detail.messages[1].role == "assistant"
    assert detail.messages[1].content == "AARYX is your adaptive AI companion."


@pytest.mark.asyncio
async def test_conversation_listing_and_pagination(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify listing conversations with pagination and updated_at ordering."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    # Create 3 conversations
    c1 = await service.create_conversation(title="Thread 1")
    await service.create_conversation(title="Thread 2")
    await service.create_conversation(title="Thread 3")

    # Add a message to Thread 1 so its updated_at becomes newest
    await service.save_message(
        conversation_id=c1.id,
        role="user",
        content="New message in thread 1",
    )

    res = await service.list_conversations(limit=2, offset=0)
    assert res.total == 3
    assert len(res.conversations) == 2
    # Thread 1 should be first because it was updated most recently
    assert res.conversations[0].id == c1.id

    # Second page
    res_page2 = await service.list_conversations(limit=2, offset=2)
    assert len(res_page2.conversations) == 1


@pytest.mark.asyncio
async def test_conversation_update(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify updating conversation title and metadata."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    conv = await service.create_conversation(title="Original Title")
    updated = await service.update_conversation(
        conversation_id=conv.id,
        title="Updated Title",
        metadata={"category": "tech"},
    )
    assert updated is not None
    assert updated.title == "Updated Title"
    assert updated.metadata_json == {"category": "tech"}


@pytest.mark.asyncio
async def test_conversation_deletion_and_cascades(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify deleting a conversation cascades and removes its messages."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    conv = await service.create_conversation(title="Thread to Delete")
    await service.save_message(conversation_id=conv.id, role="user", content="Hello")
    await service.save_message(conversation_id=conv.id, role="assistant", content="Hi!")

    deleted = await service.delete_conversation(conv.id)
    assert deleted is True

    # Conversation should no longer exist
    assert await service.get_conversation(conv.id) is None

    # Deleting nonexistent conversation returns False
    assert await service.delete_conversation("nonexistent-id") is False

    # Check database messages table directly to ensure cascade deleted them
    async with session_factory() as session:
        msg_res = await session.execute(
            text("SELECT count(*) FROM messages WHERE conversation_id = :id"),
            {"id": conv.id},
        )
        assert msg_res.scalar() == 0


@pytest.mark.asyncio
async def test_message_persistence_details(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify storing messages with provider, model, and token usage."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    conv = await service.create_conversation(title="Detailed Message Thread")
    msg = await service.save_message(
        conversation_id=conv.id,
        role="assistant",
        content="Here is the generated answer.",
        model="qwen3.5:4b",
        provider="ollama",
        token_usage={"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    )

    assert msg.id is not None
    assert msg.role == "assistant"
    assert msg.content == "Here is the generated answer."
    assert msg.model == "qwen3.5:4b"
    assert msg.provider == "ollama"
    assert msg.token_usage == {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}


@pytest.mark.asyncio
async def test_chronological_history_and_limits(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify chronological ordering and history window limits."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory, max_history_messages=4)

    conv = await service.create_conversation(title="Window Test")

    # Add 6 sequential messages
    for i in range(1, 7):
        role = "user" if i % 2 == 1 else "assistant"
        await service.save_message(
            conversation_id=conv.id,
            role=role,
            content=f"Message {i}",
        )

    # get_recent_history should apply limit=4 and return messages 3, 4, 5, 6 in ascending order
    history = await service.get_recent_history(conv.id)
    assert len(history) == 4
    assert [m.content for m in history] == ["Message 3", "Message 4", "Message 5", "Message 6"]

    # Explicit limit override
    custom_history = await service.get_recent_history(conv.id, limit=2)
    assert len(custom_history) == 2
    assert [m.content for m in custom_history] == ["Message 5", "Message 6"]


@pytest.mark.asyncio
async def test_invalid_conversation_ids(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify validation and error raising for nonexistent or empty conversation IDs."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    # Empty ID validation error
    with pytest.raises(ConversationValidationError):
        await service.get_conversation("   ")

    with pytest.raises(ConversationValidationError):
        await service.save_message(conversation_id="", role="user", content="Hello")

    with pytest.raises(ConversationValidationError):
        await service.save_message(conversation_id="conv-1", role="user", content="   ")

    # Nonexistent conversation ID raises ConversationNotFoundError on message insertion
    with pytest.raises(ConversationNotFoundError):
        await service.save_message(
            conversation_id="nonexistent-id",
            role="user",
            content="Hello",
        )

    with pytest.raises(ConversationNotFoundError):
        await service.get_recent_history(conversation_id="nonexistent-id")


@pytest.mark.asyncio
async def test_transaction_rollback(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify transaction rollback on database error."""
    _, session_factory = temp_db

    async with session_factory() as session:
        repo = ConversationRepository(session)
        conv = await repo.create_conversation(title="Rollback Test")
        conv_id = conv.id
        await session.commit()

        # Attempt to insert an invalid message with a duplicate ID to trigger rollback
        with pytest.raises(IntegrityError):
            async with session.begin():
                await repo.save_message(
                    conversation_id=conv_id,
                    role="user",
                    content="Valid message",
                    message_id="fixed-id-1",
                )
                # Force failure with duplicate primary key
                await repo.save_message(
                    conversation_id=conv_id,
                    role="user",
                    content="Duplicate message",
                    message_id="fixed-id-1",
                )

    # Verify that because transaction failed, neither message was permanently committed
    async with session_factory() as session:
        repo = ConversationRepository(session)
        messages = await repo.get_messages(conv_id)
        assert len(messages) == 0


@pytest.mark.asyncio
async def test_application_restart_persistence(tmp_path: Path) -> None:
    """Verify conversation data survives application restart by closing and reopening database."""
    db_file = tmp_path / "persistent_restart.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    # First session: Start app, create conversation, save messages
    engine1 = create_database_engine(db_url)
    await init_db(engine1)
    factory1 = get_session_factory(engine1)
    service1 = MemoryService(session_factory=factory1)

    conv = await service1.create_conversation(title="Restart Persistence Test")
    await service1.save_message(conversation_id=conv.id, role="user", content="Before restart")
    await service1.save_message(
        conversation_id=conv.id,
        role="assistant",
        content="Saved before restart",
        model="qwen3.5:4b",
        provider="ollama",
    )

    # Simulate shutdown
    await engine1.dispose()

    # Second session: Re-open the database with a fresh engine and factory
    engine2 = create_database_engine(db_url)
    await init_db(engine2)
    factory2 = get_session_factory(engine2)
    service2 = MemoryService(session_factory=factory2)

    try:
        detail = await service2.get_conversation(conv.id, include_messages=True)
        assert isinstance(detail, ConversationDetail)
        assert detail.title == "Restart Persistence Test"
        assert len(detail.messages) == 2
        assert detail.messages[0].content == "Before restart"
        assert detail.messages[1].content == "Saved before restart"
        assert detail.messages[1].provider == "ollama"

        # Continue the conversation after restart
        await service2.save_message(
            conversation_id=conv.id,
            role="user",
            content="After restart message",
        )
        updated_history = await service2.get_recent_history(conv.id)
        assert len(updated_history) == 3
        assert updated_history[2].content == "After restart message"
    finally:
        await engine2.dispose()


def test_ensure_db_directory(tmp_path: Path) -> None:
    """Verify directory auto-creation for SQLite file paths."""
    target_dir = tmp_path / "nested" / "sub" / "data"
    db_url = f"sqlite+aiosqlite:///{target_dir}/aaryx.db"

    assert not target_dir.exists()
    ensure_db_directory(db_url)
    assert target_dir.exists()

    # In-memory should be a no-op
    ensure_db_directory("sqlite+aiosqlite:///:memory:")


@pytest.mark.asyncio
async def test_sqlite_pragmas_and_wal_mode(tmp_path: Path) -> None:
    """Verify WAL mode, busy timeout, foreign keys, and synchronous mode pragmas."""
    db_file = tmp_path / "pragmas_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    assert is_file_based_sqlite(db_url) is True
    assert is_file_based_sqlite("sqlite+aiosqlite:///:memory:") is False

    engine = create_database_engine(db_url, busy_timeout_ms=5000)
    await init_db(engine)

    async with engine.connect() as conn:
        fk = (await conn.execute(text("PRAGMA foreign_keys;"))).scalar()
        bt = (await conn.execute(text("PRAGMA busy_timeout;"))).scalar()
        jm = (await conn.execute(text("PRAGMA journal_mode;"))).scalar()
        sync = (await conn.execute(text("PRAGMA synchronous;"))).scalar()

        assert fk == 1
        assert bt == 5000
        assert str(jm).lower() == "wal"
        # 1 corresponds to NORMAL in SQLite synchronous mode
        assert sync == 1

    await engine.dispose()

    # Also verify in-memory database does not fail and sets foreign keys and busy timeout
    mem_engine = create_database_engine("sqlite+aiosqlite:///:memory:")
    await init_db(mem_engine)
    async with mem_engine.connect() as conn:
        mem_fk = (await conn.execute(text("PRAGMA foreign_keys;"))).scalar()
        mem_bt = (await conn.execute(text("PRAGMA busy_timeout;"))).scalar()
        assert mem_fk == 1
        assert mem_bt == 5000
    await mem_engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_reads_and_writes(tmp_path: Path) -> None:
    """Verify concurrent reads and writes against file-backed SQLite succeed without database locks."""
    db_file = tmp_path / "concurrency_test.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_database_engine(db_url, busy_timeout_ms=5000)
    await init_db(engine)
    session_factory = get_session_factory(engine)
    service = MemoryService(session_factory=session_factory)

    # Worker to create conversation and append messages
    async def writer_worker(worker_id: int) -> str:
        conv = await service.create_conversation(title=f"Worker Thread {worker_id}")
        for i in range(3):
            await service.save_message(
                conversation_id=conv.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} from worker {worker_id}",
            )
        return conv.id

    # Worker to read conversations and lists
    async def reader_worker(iterations: int) -> int:
        total_seen = 0
        for _ in range(iterations):
            res = await service.list_conversations(limit=10, offset=0)
            total_seen += res.total
            await asyncio.sleep(0.005)
        return total_seen

    # Execute 5 writers and 5 readers concurrently under high contention
    writer_tasks = [writer_worker(i) for i in range(5)]
    reader_tasks = [reader_worker(5) for _ in range(5)]

    results = await asyncio.gather(*writer_tasks, *reader_tasks)
    created_conv_ids = results[:5]
    assert len(created_conv_ids) == 5

    # Check total conversations created
    res = await service.list_conversations(limit=20)
    assert res.total == 5

    await engine.dispose()


@pytest.mark.asyncio
async def test_conversation_listing_aggregate_count(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify SQL aggregate COUNT in list_conversations correctly counts messages without loading them."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory)

    # Conversation 1: 0 messages
    c1 = await service.create_conversation(title="Zero Messages")
    # Conversation 2: 2 messages
    c2 = await service.create_conversation(title="Two Messages")
    await service.save_message(conversation_id=c2.id, role="user", content="Hi")
    await service.save_message(conversation_id=c2.id, role="assistant", content="Hello")
    # Conversation 3: 3 messages
    c3 = await service.create_conversation(title="Three Messages")
    await service.save_message(conversation_id=c3.id, role="user", content="Q1")
    await service.save_message(conversation_id=c3.id, role="assistant", content="A1")
    await service.save_message(conversation_id=c3.id, role="user", content="Q2")

    res = await service.list_conversations(limit=10)
    assert res.total == 3

    counts = {conv.id: conv.message_count for conv in res.conversations}
    assert counts[c1.id] == 0
    assert counts[c2.id] == 2
    assert counts[c3.id] == 3


@pytest.mark.asyncio
async def test_prepare_turn_atomic_success_and_rollback(
    temp_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Verify atomic prepare_turn validates, retrieves history, and persists user prompt in one transaction."""
    _, session_factory = temp_db
    service = MemoryService(session_factory=session_factory, max_history_messages=5)

    # 1. Turn 1: Auto-create conversation
    conv_id, history1 = await service.prepare_turn(
        conversation_id=None,
        user_prompt="Hello, this is my first prompt",
        title_fallback="Initial Chat",
    )
    assert conv_id is not None
    assert len(history1) == 0

    # Assistant responds
    await service.save_message(
        conversation_id=conv_id,
        role="assistant",
        content="Hello! How can I help you?",
    )

    # 2. Turn 2: Continue existing conversation
    conv_id2, history2 = await service.prepare_turn(
        conversation_id=conv_id,
        user_prompt="Second user message",
    )
    assert conv_id2 == conv_id
    assert len(history2) == 2
    assert history2[0].content == "Hello, this is my first prompt"
    assert history2[1].content == "Hello! How can I help you?"

    # Verify messages in conversation
    messages = await service.get_messages(conv_id)
    assert len(messages) == 3
    assert messages[2].content == "Second user message"

    # 3. Invalid conversation ID raises ConversationNotFoundError and rolls back
    with pytest.raises(ConversationNotFoundError):
        await service.prepare_turn(
            conversation_id="nonexistent-uuid",
            user_prompt="Will fail",
        )

    # 4. Empty prompt raises ConversationValidationError
    with pytest.raises(ConversationValidationError):
        await service.prepare_turn(
            conversation_id=conv_id,
            user_prompt="   ",
        )
