"""SQLAlchemy ORM models for conversational memory persistence."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return current UTC timestamp with timezone information."""
    return datetime.now(UTC)


def generate_uuid() -> str:
    """Generate a standard UUID4 string."""
    return str(uuid4())


class Base(DeclarativeBase):
    """Base class for all persistent memory database models."""


class Conversation(Base):
    """Represents an independent conversation thread in AARYX memory."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        doc="Unique UUID identifier for the conversation",
    )
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Conversation",
        nullable=False,
        doc="Human-readable title or topic of the conversation",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
        doc="Timestamp when the conversation was initiated",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        index=True,
        doc="Timestamp of the most recent activity in this conversation",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Extensible metadata dictionary for tags, settings, or client info",
    )

    # Cascading relation to messages.
    # Never loaded implicitly: a conversation can hold thousands of messages, and most callers
    # only need the conversation row (existence checks, titles, counts). Callers that need the
    # thread load it explicitly (selectinload) or query it with a LIMIT through the repository.
    # `passive_deletes` lets the ORM delete a conversation without first loading its messages.
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at.asc()",
        lazy="raise",
    )


class Message(Base):
    """Represents a single message entry within a conversation thread."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        doc="Unique UUID identifier for the message",
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Foreign key linking to parent conversation",
    )
    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Role of the sender: user, assistant, or system",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The text body of the message",
    )
    model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        doc="Model identifier that generated the response (assistant messages)",
    )
    provider: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        doc="Provider that generated the response (e.g., ollama, openai)",
    )
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc="Token usage breakdown if reported",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
        doc="Timestamp when the message was recorded",
    )

    # Relationship back to conversation
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )


# Composite index for optimized chronological history fetching
Index("ix_messages_conv_created", Message.conversation_id, Message.created_at)


class Memory(Base):
    """Represents a discrete, long-term fact, preference, or user instruction.

    Persists across conversations independently of individual chat threads.
    """

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        doc="Unique UUID identifier for the memory entry",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="The text body of the learned fact, preference, or instruction statement",
    )
    category: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="fact",
        doc="Category: preference, fact, project, biography, instruction",
    )
    origin: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="explicit",
        doc="Extraction origin: explicit (direct user command) or inferred (deduced by LLM)",
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        doc="Confidence score from 0.0 to 1.0 (1.0 for explicit, 0.0-1.0 for inferred)",
    )
    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=3.0,
        doc="Priority weighting from 1.0 (low) to 5.0 (critical core constraint)",
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
        index=True,
        doc="Lifecycle state: candidate, active, superseded, deleted",
    )
    superseded_by_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("memories.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Real foreign key pointing to the newer memory that replaced this record",
    )
    source_conversation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Optional conversation where this memory originated; preserved on conversation deletion",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
        doc="Timestamp when the memory record was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
        index=True,
        doc="Timestamp when the memory record was last updated",
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when this memory was last retrieved for prompt context injection",
    )
    access_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Total number of times this memory was retrieved for prompt context injection",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        doc="Extensible metadata dictionary for tags, conflict reasons, or extraction context",
    )

    # Relationships
    superseded_by: Mapped["Memory | None"] = relationship(
        "Memory",
        remote_side=[id],
        foreign_keys=[superseded_by_id],
        lazy="raise",
    )
    source_conversation: Mapped["Conversation | None"] = relationship(
        "Conversation",
        foreign_keys=[source_conversation_id],
        lazy="raise",
    )

    __table_args__ = (
        Index("ix_memories_status_category", "status", "category"),
        Index("ix_memories_status_created_at", "status", "created_at"),
    )
