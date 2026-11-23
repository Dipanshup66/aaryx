"""Repository encapsulating database operations for conversations and messages."""

from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.memory.models import Conversation, Memory, Message, generate_uuid, utc_now

# Metadata keys written on a memory whose superseding memory was permanently deleted.
SUPERSEDED_BY_DELETED_KEY = "superseded_by_deleted"
FORMER_SUCCESSOR_KEY = "former_superseded_by_id"


class ConversationRepository:
    """Async repository providing CRUD operations for conversations and messages."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with an active AsyncSession."""
        self.session = session

    async def create_conversation(
        self,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> Conversation:
        """Create and persist a new conversation thread."""
        conv = Conversation(
            id=conversation_id or generate_uuid(),
            title=title or "New Conversation",
            metadata_json=metadata if metadata is not None else {},
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        self.session.add(conv)
        await self.session.flush()
        return conv

    async def get_conversation(
        self,
        conversation_id: str,
        include_messages: bool = False,
    ) -> Conversation | None:
        """Retrieve a conversation by its unique identifier."""
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        if include_messages:
            stmt = stmt.options(selectinload(Conversation.messages))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_messages(self, conversation_id: str) -> int:
        """Count a conversation's messages with SQL COUNT (never loads message rows)."""
        stmt = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
        return int((await self.session.execute(stmt)).scalar() or 0)

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[tuple[Conversation, int]], int]:
        """List conversations ordered by most recent activity, with total count.

        Uses SQL aggregate COUNT on Message to avoid loading message entities into memory.
        """
        count_stmt = select(func.count(Conversation.id))
        total = (await self.session.execute(count_stmt)).scalar() or 0

        stmt = (
            select(Conversation, func.count(Message.id).label("message_count"))
            .outerjoin(Message, Message.conversation_id == Conversation.id)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        rows = list(result.all())
        conversations_with_counts = [(conv, int(count)) for conv, count in rows]
        return conversations_with_counts, total

    async def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation | None:
        """Update an existing conversation's title or metadata."""
        conv = await self.get_conversation(conversation_id, include_messages=False)
        if not conv:
            return None

        if title is not None:
            conv.title = title
        if metadata is not None:
            conv.metadata_json = metadata
        conv.updated_at = utc_now()

        await self.session.flush()
        return conv

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and cascade-delete its messages.

        Returns True if deleted, False if conversation was not found.
        """
        conv = await self.get_conversation(conversation_id, include_messages=False)
        if not conv:
            return False

        # Remove messages with one statement instead of loading them for ORM cascade. This does
        # not depend on SQLite's foreign_keys pragma being enabled on the connection.
        await self.session.execute(
            delete(Message).where(Message.conversation_id == conversation_id)
        )
        await self.session.delete(conv)
        await self.session.flush()
        return True

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        provider: str | None = None,
        token_usage: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> Message:
        """Save a new message associated with a conversation and update conversation timestamp."""
        msg = Message(
            id=message_id or generate_uuid(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            provider=provider,
            token_usage=token_usage,
            created_at=utc_now(),
        )
        self.session.add(msg)

        # Touch parent conversation's updated_at directly without redundant SELECT
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=utc_now())
        )

        await self.session.flush()
        return msg

    async def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """Retrieve all or a slice of messages in chronological order."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        if offset:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20,
    ) -> list[Message]:
        """Retrieve the latest N messages in ascending chronological order."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        messages = list(result.scalars().all())
        messages.reverse()
        return messages


class MemoryRepository:
    """Async repository providing CRUD operations for long-term memories."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with an active AsyncSession."""
        self.session = session

    async def create_memory(
        self,
        content: str,
        category: str = "fact",
        origin: str = "explicit",
        confidence: float = 1.0,
        importance: float = 3.0,
        status: str = "active",
        superseded_by_id: str | None = None,
        source_conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        """Create and persist a new long-term memory record."""
        memory = Memory(
            id=memory_id or generate_uuid(),
            content=content,
            category=category,
            origin=origin,
            confidence=confidence,
            importance=importance,
            status=status,
            superseded_by_id=superseded_by_id,
            source_conversation_id=source_conversation_id,
            metadata_json=metadata if metadata is not None else {},
            created_at=utc_now(),
            updated_at=utc_now(),
            last_accessed_at=None,
            access_count=0,
        )
        self.session.add(memory)
        await self.session.flush()
        return memory

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by its unique identifier."""
        stmt = select(Memory).where(Memory.id == memory_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Memory], int]:
        """List memories matching status and category filters with total count.

        By default, soft-deleted memories are excluded unless explicitly requested
        or include_deleted is True.
        """
        filters = []
        if status is not None:
            filters.append(Memory.status == status)
        elif not include_deleted:
            filters.append(Memory.status != "deleted")

        if category is not None:
            filters.append(Memory.category == category)

        count_stmt = select(func.count(Memory.id))
        if filters:
            count_stmt = count_stmt.where(*filters)
        total = (await self.session.execute(count_stmt)).scalar() or 0

        stmt = select(Memory)
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.order_by(Memory.created_at.desc(), Memory.id.desc()).offset(offset).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all()), total

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        importance: float | None = None,
        status: str | None = None,
        superseded_by_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update an existing memory record without modifying access counters."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return None

        if content is not None:
            memory.content = content
        if category is not None:
            memory.category = category
        if confidence is not None:
            memory.confidence = confidence
        if importance is not None:
            memory.importance = importance
        if status is not None:
            memory.status = status
        if superseded_by_id is not None:
            memory.superseded_by_id = superseded_by_id
        if metadata is not None:
            memory.metadata_json = metadata

        memory.updated_at = utc_now()
        await self.session.flush()
        return memory

    async def delete_memory(
        self,
        memory_id: str,
        hard: bool = False,
    ) -> tuple[bool, Memory | None]:
        """Delete a memory record.

        Defaults to soft delete (setting status='deleted').
        Hard deletion permanently removes the record from the database.
        Returns (success, memory_instance).
        """
        memory = await self.get_memory(memory_id)
        if not memory:
            return False, None

        if hard:
            await self._detach_superseded_predecessors(memory)
            await self.session.delete(memory)
        else:
            memory.status = "deleted"
            memory.updated_at = utc_now()

        await self.session.flush()
        return True, memory

    async def _detach_superseded_predecessors(self, memory: Memory) -> int:
        """Keep supersession lineage truthful before ``memory`` is purged.

        Memories that were superseded BY ``memory`` must not silently lose that link when the
        row disappears (the foreign key alone would just null it, leaving a superseded memory
        with no recorded successor). Their status is never touched, so nothing is reactivated.

        * If ``memory`` was itself superseded by a live memory, predecessors are re-linked
          to that memory (chain A -> B -> C, purge B  =>  A -> C).
        * Otherwise predecessors get an explicit tombstone in their metadata
          (``superseded_by_deleted`` plus ``former_superseded_by_id``).

        Returns the number of predecessors updated.
        """
        stmt = select(Memory).where(Memory.superseded_by_id == memory.id, Memory.id != memory.id)
        predecessors = list((await self.session.execute(stmt)).scalars().all())
        successor_id = memory.superseded_by_id

        for pred in predecessors:
            if successor_id and successor_id != pred.id:
                pred.superseded_by_id = successor_id
            else:
                pred.superseded_by_id = None
                pred.metadata_json = {
                    **(pred.metadata_json or {}),
                    SUPERSEDED_BY_DELETED_KEY: True,
                    FORMER_SUCCESSOR_KEY: memory.id,
                }
            pred.updated_at = utc_now()

        if predecessors:
            await self.session.flush()
        return len(predecessors)

    async def touch_memory_access(self, memory_id: str) -> None:
        """Update last_accessed_at and access_count when retrieved for prompt context injection."""
        stmt = (
            update(Memory)
            .where(Memory.id == memory_id)
            .values(
                last_accessed_at=utc_now(),
                access_count=Memory.access_count + 1,
            )
        )
        await self.session.execute(stmt)
        await self.session.flush()
