"""Memory service coordinating repository transactions, business validation, and AI context."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.memory.exceptions import (
    ConversationNotFoundError,
    ConversationValidationError,
    MemoryNotFoundError,
    MemoryValidationError,
)
from app.memory.extractor import (
    ExplicitMemoryParser,
    InferredMemoryExtractor,
    detect_contradiction,
    find_exact_duplicate,
    stemmed_tokens,
)
from app.memory.models import Memory
from app.memory.repository import ConversationRepository, MemoryRepository
from app.memory.schemas import (
    ConversationDetail,
    ConversationListResponse,
    ConversationRead,
    MemoryApprove,
    MemoryCreate,
    MemoryDeleteResponse,
    MemoryListResponse,
    MemoryRead,
    MemoryUpdate,
    MessageRead,
    normalize_memory_content,
)
from app.providers.base import ChatMessage

logger = get_logger("memory.service")


class MemoryService:
    """Coordinates conversation storage, history retrieval, and lifecycle management."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        max_history_messages: int = 20,
        settings: Settings | None = None,
        extractor: InferredMemoryExtractor | None = None,
    ) -> None:
        """Initialize MemoryService with session maker, history limits, and extraction engine."""
        self.session_factory = session_factory
        self.max_history_messages = max_history_messages
        self.settings = settings or get_settings()
        self.explicit_parser = ExplicitMemoryParser
        self.extractor = extractor or InferredMemoryExtractor(settings=self.settings)
        # Serializes memory read-modify-write so parallel turns cannot double-insert a suggestion.
        self._memory_write_lock = asyncio.Lock()

    async def create_conversation(
        self,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
    ) -> ConversationRead:
        """Create a new independent conversation thread."""
        async with self.session_factory() as session, session.begin():
            repo = ConversationRepository(session)
            conv = await repo.create_conversation(
                title=title,
                metadata=metadata,
                conversation_id=conversation_id,
            )
            logger.info("Created conversation %s (title='%s')", conv.id, conv.title)
            return ConversationRead(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                metadata_json=conv.metadata_json,
                message_count=0,
            )

    async def get_conversation(
        self,
        conversation_id: str,
        include_messages: bool = False,
    ) -> ConversationDetail | ConversationRead | None:
        """Retrieve a conversation summary or detailed thread with all messages."""
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")

        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(
                conversation_id,
                include_messages=include_messages,
            )
            if not conv:
                return None

            if include_messages:
                messages = [
                    MessageRead(
                        id=m.id,
                        conversation_id=m.conversation_id,
                        role=m.role,  # type: ignore[arg-type]
                        content=m.content,
                        model=m.model,
                        provider=m.provider,
                        token_usage=m.token_usage,
                        created_at=m.created_at,
                    )
                    for m in conv.messages
                ]
                return ConversationDetail(
                    id=conv.id,
                    title=conv.title,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    metadata_json=conv.metadata_json,
                    message_count=len(messages),
                    messages=messages,
                )

            return ConversationRead(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                metadata_json=conv.metadata_json,
                message_count=await repo.count_messages(conv.id),
            )

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> ConversationListResponse:
        """List active conversations ordered by recent activity."""
        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv_rows, total = await repo.list_conversations(limit=limit, offset=offset)

            items = [
                ConversationRead(
                    id=c.id,
                    title=c.title,
                    created_at=c.created_at,
                    updated_at=c.updated_at,
                    metadata_json=c.metadata_json,
                    message_count=count,
                )
                for c, count in conv_rows
            ]

            return ConversationListResponse(
                conversations=items,
                total=total,
                limit=limit,
                offset=offset,
            )

    async def update_conversation(
        self,
        conversation_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationRead | None:
        """Update conversation title or metadata attributes."""
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")

        async with self.session_factory() as session, session.begin():
            repo = ConversationRepository(session)
            conv = await repo.update_conversation(
                conversation_id=conversation_id,
                title=title,
                metadata=metadata,
            )
            if not conv:
                return None

            return ConversationRead(
                id=conv.id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                metadata_json=conv.metadata_json,
                message_count=await repo.count_messages(conv.id),
            )

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation and all cascaded messages."""
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")

        async with self.session_factory() as session, session.begin():
            repo = ConversationRepository(session)
            deleted = await repo.delete_conversation(conversation_id)
            if deleted:
                logger.info("Deleted conversation %s and associated messages", conversation_id)
            return deleted

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        provider: str | None = None,
        token_usage: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> MessageRead:
        """Persist a message within an existing conversation."""
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")
        if not content or not content.strip():
            raise ConversationValidationError("Message content cannot be empty.")

        async with self.session_factory() as session, session.begin():
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if not conv:
                raise ConversationNotFoundError(conversation_id)

            msg = await repo.save_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                model=model,
                provider=provider,
                token_usage=token_usage,
                message_id=message_id,
            )

            logger.info(
                "Stored %s message in conversation %s (model=%s, provider=%s)",
                role,
                conversation_id,
                model or "n/a",
                provider or "n/a",
            )

            return MessageRead(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,  # type: ignore[arg-type]
                content=msg.content,
                model=msg.model,
                provider=msg.provider,
                token_usage=msg.token_usage,
                created_at=msg.created_at,
            )

    async def get_messages(
        self,
        conversation_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MessageRead]:
        """Retrieve stored messages for a conversation."""
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")

        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if not conv:
                raise ConversationNotFoundError(conversation_id)

            messages = await repo.get_messages(
                conversation_id=conversation_id,
                limit=limit,
                offset=offset,
            )
            return [
                MessageRead(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    role=m.role,  # type: ignore[arg-type]
                    content=m.content,
                    model=m.model,
                    provider=m.provider,
                    token_usage=m.token_usage,
                    created_at=m.created_at,
                )
                for m in messages
            ]

    async def get_recent_history(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        """Retrieve recent conversation history formatted for BrainOrchestrator.

        Enforces chronological order and limits message count.
        """
        if not conversation_id or not conversation_id.strip():
            raise ConversationValidationError("Conversation ID cannot be empty.")

        max_limit = limit if limit is not None else self.max_history_messages

        async with self.session_factory() as session:
            repo = ConversationRepository(session)
            conv = await repo.get_conversation(conversation_id)
            if not conv:
                raise ConversationNotFoundError(conversation_id)

            messages = await repo.get_recent_messages(
                conversation_id=conversation_id,
                limit=max_limit,
            )

            return [
                ChatMessage(
                    role=m.role,  # type: ignore[arg-type]
                    content=m.content,
                )
                for m in messages
            ]

    async def prepare_turn(
        self,
        conversation_id: str | None,
        user_prompt: str,
        title_fallback: str | None = None,
    ) -> tuple[str, list[ChatMessage]]:
        """Atomically validate/create conversation, fetch history window, and persist user prompt.

        Executes within a single session and transaction to eliminate redundant connections
        and guarantee ACID safety before model inference begins.

        Returns:
            tuple[active_conversation_id, recent_history_messages]
        """
        if not user_prompt or not user_prompt.strip():
            raise ConversationValidationError("Message content cannot be empty.")

        async with self.session_factory() as session, session.begin():
            repo = ConversationRepository(session)

            if conversation_id:
                if not conversation_id.strip():
                    raise ConversationValidationError("Conversation ID cannot be empty.")
                conv = await repo.get_conversation(conversation_id, include_messages=False)
                if not conv:
                    raise ConversationNotFoundError(conversation_id)

                messages = await repo.get_recent_messages(
                    conversation_id=conversation_id,
                    limit=self.max_history_messages,
                )
                history = [
                    ChatMessage(
                        role=m.role,  # type: ignore[arg-type]
                        content=m.content,
                    )
                    for m in messages
                ]
                active_id = conv.id
            else:
                title = title_fallback or user_prompt[:50].strip()
                conv = await repo.create_conversation(title=title)
                active_id = conv.id
                history = []

            # Persist user prompt prior to inference
            await repo.save_message(
                conversation_id=active_id,
                role="user",
                content=user_prompt,
            )

            logger.info(
                "Atomically prepared turn for conversation %s (history_count=%d)",
                active_id,
                len(history),
            )
            return active_id, history

    # -------------------------------------------------------------------------
    # Long-Term Memory CRUD Operations
    # -------------------------------------------------------------------------

    async def create_memory(self, payload: MemoryCreate) -> MemoryRead:
        """Create and persist a new long-term memory record."""
        async with self.session_factory() as session, session.begin():
            # Validate source conversation existence if supplied
            if payload.source_conversation_id:
                conv_repo = ConversationRepository(session)
                conv = await conv_repo.get_conversation(payload.source_conversation_id)
                if not conv:
                    raise MemoryValidationError(
                        f"Referenced source conversation '{payload.source_conversation_id}' does not exist."
                    )

            repo = MemoryRepository(session)
            mem = await repo.create_memory(
                content=payload.content,
                category=payload.category,
                origin=payload.origin,
                confidence=payload.confidence,
                importance=payload.importance,
                status=payload.status,
                source_conversation_id=payload.source_conversation_id,
                metadata=payload.metadata,
            )
            logger.info(
                "Created long-term memory %s (category='%s', origin='%s', status='%s')",
                mem.id,
                mem.category,
                mem.origin,
                mem.status,
            )
            return MemoryRead.model_validate(mem)

    async def get_memory(self, memory_id: str) -> MemoryRead | None:
        """Retrieve a specific long-term memory record by its identifier."""
        if not memory_id or not memory_id.strip():
            raise MemoryValidationError("Memory ID cannot be empty.")

        async with self.session_factory() as session:
            repo = MemoryRepository(session)
            mem = await repo.get_memory(memory_id.strip())
            if not mem:
                return None
            return MemoryRead.model_validate(mem)

    async def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> MemoryListResponse:
        """Retrieve paginated memories matching status and category filters."""
        async with self.session_factory() as session:
            repo = MemoryRepository(session)
            items, total = await repo.list_memories(
                status=status,
                category=category,
                include_deleted=include_deleted,
                limit=limit,
                offset=offset,
            )
            return MemoryListResponse(
                memories=[MemoryRead.model_validate(m) for m in items],
                total=total,
                limit=limit,
                offset=offset,
            )

    async def update_memory(self, memory_id: str, payload: MemoryUpdate) -> MemoryRead:
        """Update an existing long-term memory record with invariant enforcement."""
        if not memory_id or not memory_id.strip():
            raise MemoryValidationError("Memory ID cannot be empty.")

        mid = memory_id.strip()

        async with self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            existing = await repo.get_memory(mid)
            if not existing:
                raise MemoryNotFoundError(mid)

            # Invariant 6: Soft-deleted memories cannot be modified unless explicitly reactivated
            if existing.status == "deleted" and payload.status != "active":
                raise MemoryValidationError(
                    f"Memory '{mid}' is soft-deleted and cannot be modified. To reactivate, explicitly set status to 'active'."
                )

            # Invariant 3 & 4: Superseded status and non-circular superseded_by_id validation
            target_status = payload.status if payload.status is not None else existing.status
            target_superseded_by_id = (
                payload.superseded_by_id
                if payload.superseded_by_id is not None
                else existing.superseded_by_id
            )

            if payload.superseded_by_id is not None:
                sub_id = payload.superseded_by_id.strip()
                if sub_id == mid:
                    raise MemoryValidationError("A memory record cannot supersede itself.")

                target = await repo.get_memory(sub_id)
                if not target:
                    raise MemoryValidationError(
                        f"Superseding memory '{payload.superseded_by_id}' not found."
                    )

                # Check circular reference: traverse target's superseded_by_id chain
                curr = target
                visited = {mid, sub_id}
                while curr and curr.superseded_by_id:
                    if curr.superseded_by_id == mid:
                        raise MemoryValidationError(
                            f"Circular superseding relationship detected: memory '{sub_id}' already references '{mid}'."
                        )
                    if curr.superseded_by_id in visited:
                        break
                    visited.add(curr.superseded_by_id)
                    curr = await repo.get_memory(curr.superseded_by_id)

                if payload.status is None:
                    target_status = "superseded"

            # The pointer is required to ENTER the superseded state. Editing a memory that is
            # already superseded must stay possible even if its successor was later purged
            # (the pointer is cleared / tombstoned by the repository), and the API can never
            # clear an existing pointer.
            entering_superseded = target_status == "superseded" and existing.status != "superseded"
            if entering_superseded and not target_superseded_by_id:
                raise MemoryValidationError(
                    "A memory cannot enter the superseded state without a valid superseded_by_id relationship."
                )

            # Invariant 1 & 2: Origin and confidence consistency on update
            final_origin = existing.origin
            final_confidence = (
                payload.confidence if payload.confidence is not None else existing.confidence
            )

            # Decision D-001: an inferred memory becomes active only through explicit approval.
            if (
                final_origin == "inferred"
                and target_status == "active"
                and (existing.status != "active")
            ):
                raise MemoryValidationError(
                    "An inferred memory cannot be activated through an update; approve the "
                    "suggestion instead."
                )
            if payload.status == "rejected":
                raise MemoryValidationError("Use the reject action to reject a memory suggestion.")
            if final_origin == "explicit" and final_confidence < 0.90:
                raise MemoryValidationError(
                    "Explicit memories must have a high confidence score (>= 0.90)."
                )

            updated = await repo.update_memory(
                memory_id=mid,
                content=payload.content,
                category=payload.category,
                confidence=payload.confidence,
                importance=payload.importance,
                status=target_status,
                superseded_by_id=payload.superseded_by_id,
                metadata=payload.metadata,
            )
            if not updated:
                raise MemoryNotFoundError(mid)

            logger.info("Updated long-term memory %s (status='%s')", mid, updated.status)
            return MemoryRead.model_validate(updated)

    async def get_context_memories(self, query: str, limit: int | None = None) -> list[MemoryRead]:
        """Select the active memories worth adding to the prompt for this user message.

        Only ``active`` memories are ever returned: suggestions, rejected, superseded and
        deleted memories never reach a prompt. Selection is deterministic and local (no
        embeddings): core memories (instructions and anything with importance >= 4, which
        includes everything the user explicitly asked to remember) are always candidates, and
        other memories qualify when they share meaningful words with the user's message.
        Selected memories have their access counters updated.
        """
        max_items = self.settings.memory_max_context_items if limit is None else limit
        if max_items <= 0 or not query or not query.strip():
            return []

        async with self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            active = await self._load_all_by_status(repo, "active")
            selected = self._rank_for_context(query, active)[:max_items]
            for mem in selected:
                await repo.touch_memory_access(mem.id)
            return [MemoryRead.model_validate(m) for m in selected]

    @staticmethod
    def _rank_for_context(query: str, active_memories: list[Memory]) -> list[Memory]:
        """Order memories by relevance to ``query``, dropping the ones that do not qualify."""
        query_tokens = stemmed_tokens(normalize_memory_content(query))

        def score(mem: Memory) -> float:
            overlap = len(query_tokens & stemmed_tokens(normalize_memory_content(mem.content)))
            is_core = mem.category == "instruction" or mem.importance >= 4.0
            if overlap == 0 and not is_core:
                return -1.0
            return overlap * 2 + mem.importance / 5 + (1.0 if mem.category == "instruction" else 0)

        scored = [(score(m), m) for m in active_memories]
        qualified = [(s, m) for s, m in scored if s >= 0]
        qualified.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        unique: list[Memory] = []
        seen: set[str] = set()
        for _, mem in qualified:  # identical text is only worth one prompt line
            key = normalize_memory_content(mem.content)
            if key not in seen:
                seen.add(key)
                unique.append(mem)
        return unique

    async def approve_memory(
        self, memory_id: str, payload: MemoryApprove | None = None
    ) -> MemoryRead:
        """Approve a pending memory suggestion, making it active (Decision D-001).

        The user may supply corrected content. If the suggestion was held because it conflicted
        with an active memory, approving it is an explicit user decision and supersedes that
        memory.
        """
        if not memory_id or not memory_id.strip():
            raise MemoryValidationError("Memory ID cannot be empty.")

        mid = memory_id.strip()
        async with self._memory_write_lock, self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            existing = await repo.get_memory(mid)
            if not existing:
                raise MemoryNotFoundError(mid)
            if existing.status != "candidate":
                raise MemoryValidationError(
                    f"Memory '{mid}' is not a pending suggestion (status='{existing.status}')."
                )

            metadata = {
                **(existing.metadata_json or {}),
                "approved_by_user": True,
                "approved_at": datetime.now(UTC).isoformat(),
            }
            conflicts_with = metadata.get("conflicts_with")

            updated = await repo.update_memory(
                memory_id=mid,
                content=payload.content if payload else None,
                status="active",
                metadata=metadata,
            )
            if not updated:
                raise MemoryNotFoundError(mid)

            if isinstance(conflicts_with, str) and conflicts_with != mid:
                older = await repo.get_memory(conflicts_with)
                if older is not None and older.status == "active":
                    await repo.update_memory(
                        memory_id=older.id,
                        status="superseded",
                        superseded_by_id=mid,
                    )

            logger.info("Approved memory suggestion %s", mid)
            return MemoryRead.model_validate(updated)

    async def reject_memory(self, memory_id: str) -> MemoryRead:
        """Reject a pending memory suggestion. Rejected suggestions are never used or re-suggested."""
        if not memory_id or not memory_id.strip():
            raise MemoryValidationError("Memory ID cannot be empty.")

        mid = memory_id.strip()
        async with self._memory_write_lock, self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            existing = await repo.get_memory(mid)
            if not existing:
                raise MemoryNotFoundError(mid)
            if existing.status != "candidate":
                raise MemoryValidationError(
                    f"Memory '{mid}' is not a pending suggestion (status='{existing.status}')."
                )

            updated = await repo.update_memory(
                memory_id=mid,
                status="rejected",
                metadata={
                    **(existing.metadata_json or {}),
                    "rejected_by_user": True,
                    "rejected_at": datetime.now(UTC).isoformat(),
                },
            )
            if not updated:
                raise MemoryNotFoundError(mid)

            logger.info("Rejected memory suggestion %s", mid)
            return MemoryRead.model_validate(updated)

    async def delete_memory(self, memory_id: str, hard: bool = False) -> MemoryDeleteResponse:
        """Delete a long-term memory record (defaults to soft delete)."""
        if not memory_id or not memory_id.strip():
            raise MemoryValidationError("Memory ID cannot be empty.")

        mid = memory_id.strip()
        async with self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            success, mem = await repo.delete_memory(mid, hard=hard)
            if not success or not mem:
                raise MemoryNotFoundError(mid)

            logger.info("Deleted long-term memory %s (hard=%s)", mid, hard)
            msg = f"Memory '{mid}' permanently purged." if hard else f"Memory '{mid}' soft-deleted."
            return MemoryDeleteResponse(
                id=mid,
                deleted=True,
                hard=hard,
                message=msg,
            )

    # -------------------------------------------------------------------------
    # Selective Memory Extraction, Deduplication & Contradiction Resolution
    # -------------------------------------------------------------------------

    async def extract_and_store_memories(
        self,
        user_prompt: str,
        assistant_reply: str | None = None,
        conversation_id: str | None = None,
    ) -> list[MemoryRead]:
        """Extract durable long-term memories from user input and store them safely.

        Applies explicit directive parsing, local LLM inferred extraction,
        normalized exact deduplication, and conservative contradiction resolution.
        Never modifies access_count or last_accessed_at.
        """
        if not user_prompt or not user_prompt.strip():
            return []

        # Step 1: Check for explicit memory directive first
        explicit_candidate = self.explicit_parser.parse(user_prompt)
        if explicit_candidate is not None:
            candidates = [explicit_candidate]
        elif not self.settings.memory_suggestions_enabled:
            # Suggestions are off: only an explicit "remember this" may create a memory.
            return []
        else:
            # Step 2: Fall back to local inferred memory suggestions (always 'candidate')
            candidates = await self.extractor.extract(user_prompt, assistant_reply)

        if not candidates:
            return []

        stored_memories: list[MemoryRead] = []

        async with self._memory_write_lock, self.session_factory() as session, session.begin():
            repo = MemoryRepository(session)
            # Complete (paginated) view of active memories, kept current as candidates are
            # applied so later candidates see rows created or superseded earlier in this batch.
            active_memories = await self._load_all_by_status(repo, "active")

            # Suggestions the user already rejected are never proposed again.
            # Identical suggestions that are still pending are not stacked up either.
            skip_contents: set[str] = set()
            if any(c.origin == "inferred" for c in candidates):
                for status in ("rejected", "candidate"):
                    skip_contents |= {
                        normalize_memory_content(m.content)
                        for m in await self._load_all_by_status(repo, status)
                    }

            for candidate in candidates:
                normalized = normalize_memory_content(candidate.content)
                if candidate.origin == "inferred" and normalized in skip_contents:
                    logger.info("Skipped a memory suggestion that is rejected or already pending")
                    continue
                stored = await self._apply_candidate(
                    repo, candidate, active_memories, conversation_id
                )
                stored_memories.append(stored)
                if stored.status == "candidate":
                    skip_contents.add(normalized)

        return stored_memories

    _ACTIVE_PAGE_SIZE = 500

    async def _load_all_by_status(self, repo: MemoryRepository, status: str) -> list[Memory]:
        """Load every memory with the given status, paging past the per-query limit."""
        found: list[Memory] = []
        offset = 0
        while True:
            page, total = await repo.list_memories(
                status=status, limit=self._ACTIVE_PAGE_SIZE, offset=offset
            )
            found.extend(page)
            offset += len(page)
            if not page or offset >= total:
                return found

    @staticmethod
    def _should_supersede(candidate: MemoryCreate, conflicting: Memory) -> bool:
        """Decide whether a candidate may replace the memory it contradicts.

        * A pending suggestion (status 'candidate') never replaces anything: only an approval by
          the user can supersede an active memory.
        * An inferred candidate never overwrites an explicit memory.
        * An explicit candidate always supersedes (a direct user instruction).
        * Inferred vs inferred: only high-confidence (>= 0.85) over low-importance (<= 3.0).
        """
        if candidate.status != "active":
            return False
        if candidate.origin == "explicit":
            return True
        if conflicting.origin == "explicit":
            return False
        return candidate.confidence >= 0.85 and conflicting.importance <= 3.0

    async def _apply_candidate(
        self,
        repo: MemoryRepository,
        candidate: MemoryCreate,
        active_memories: list[Memory],
        conversation_id: str | None,
    ) -> MemoryRead:
        """Store one candidate (dedupe / supersede / hold / create) and update ``active_memories``."""
        # Step A: Exact normalized deduplication check against active memories
        dup = find_exact_duplicate(candidate.content, active_memories)
        if dup is not None:
            # Update confidence and importance only if incoming evidence is higher
            new_conf = max(dup.confidence, candidate.confidence)
            new_imp = max(dup.importance, candidate.importance)
            # Note: access counters are NOT incremented during deduplication
            updated = await repo.update_memory(
                memory_id=dup.id,
                confidence=new_conf,
                importance=new_imp,
            )
            logger.info(
                "Deduplicated memory '%s' refreshed (conf=%.2f, imp=%.2f)",
                dup.id,
                new_conf,
                new_imp,
            )
            return MemoryRead.model_validate(updated or dup)

        # Step B: Conservative contradiction detection
        conflicting_mem, reason = detect_contradiction(candidate, active_memories)
        if conflicting_mem is None:
            # Step C: Standard creation (no duplicate and no contradiction)
            mem = await repo.create_memory(
                content=candidate.content,
                category=candidate.category,
                origin=candidate.origin,
                confidence=candidate.confidence,
                importance=candidate.importance,
                status=candidate.status,
                source_conversation_id=conversation_id,
                metadata=candidate.metadata,
            )
            logger.info(
                "Persisted new memory '%s' (origin='%s', status='%s')",
                mem.id,
                mem.origin,
                mem.status,
            )
            self._track_active(active_memories, mem)
            return MemoryRead.model_validate(mem)

        if self._should_supersede(candidate, conflicting_mem):
            new_mem = await repo.create_memory(
                content=candidate.content,
                category=candidate.category,
                origin=candidate.origin,
                confidence=candidate.confidence,
                importance=candidate.importance,
                status=candidate.status,
                source_conversation_id=conversation_id,
                metadata=candidate.metadata,
            )
            await repo.update_memory(
                memory_id=conflicting_mem.id,
                status="superseded",
                superseded_by_id=new_mem.id,
            )
            logger.info(
                "%s memory '%s' superseded %s memory '%s' (%s)",
                candidate.origin.capitalize(),
                new_mem.id,
                conflicting_mem.origin,
                conflicting_mem.id,
                reason,
            )
            # Keep the in-memory view consistent: the old row is no longer active.
            active_memories[:] = [m for m in active_memories if m.id != conflicting_mem.id]
            self._track_active(active_memories, new_mem)
            return MemoryRead.model_validate(new_mem)

        # Hold the incoming memory as a candidate with conflict metadata.
        # Invariant: an inferred memory must NEVER silently overwrite an explicit one.
        candidate_metadata = {
            **(candidate.metadata or {}),
            "conflict_detected": True,
            "conflicts_with": conflicting_mem.id,
            "conflict_reason": reason,
        }
        held = await repo.create_memory(
            content=candidate.content,
            category=candidate.category,
            origin=candidate.origin,
            confidence=candidate.confidence,
            importance=candidate.importance,
            status="candidate",
            source_conversation_id=conversation_id,
            metadata=candidate_metadata,
        )
        logger.info(
            "Inferred candidate '%s' conflicts with %s memory '%s'; held as candidate",
            held.id,
            conflicting_mem.origin,
            conflicting_mem.id,
        )
        return MemoryRead.model_validate(held)

    @staticmethod
    def _track_active(active_memories: list[Memory], memory: Memory) -> None:
        """Add a newly created memory to the working set if it is active."""
        if memory.status == "active":
            active_memories.insert(0, memory)
