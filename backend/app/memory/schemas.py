"""Pydantic schemas for conversation persistence and memory management."""

import re
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MessageRole = Literal["user", "assistant", "system"]

_SECURITY_OVERRIDE_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:in\s+)?developer\s+mode\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+dan\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"<(?:system|prompt_override)>", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bbypass\s+(?:all\s+)?security\b", re.IGNORECASE),
    re.compile(r"\bdisable\s+(?:all\s+)?safety\b", re.IGNORECASE),
]

_FORBIDDEN_METADATA_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "passwd",
    "token",
    "access_token",
    "auth_token",
    "private_key",
    "bearer",
}

_SENSITIVE_VALUE_PREFIXES = (
    "sk-",
    "ghp_",
    "bearer ",
    "-----begin private key-----",
    "-----begin rsa private key-----",
)

_JWT_PATTERN = re.compile(r"\bey[A-Za-z0-9_-]{15,}\.ey[A-Za-z0-9_-]{15,}")
_AWS_KEY_PATTERN = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_INDIAN_IDENTIFIER_PATTERNS = [
    re.compile(r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b"),  # Aadhaar format with spaces
    re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),  # PAN card format
]

_PASSWORD_PIN_PATTERNS = [
    re.compile(r"\b(?:password|passwd|pin(?:\s+code)?)\s*(?:is|:|=)\s*\S+", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:password|pin)\s+is\b", re.IGNORECASE),
]

_CREDENTIAL_PATTERNS = [
    re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{15,}|ghp_[a-zA-Z0-9]{20,}|gho_[a-zA-Z0-9]{20,})\b"),
    _JWT_PATTERN,
    _AWS_KEY_PATTERN,
    re.compile(r"\bbearer\s+[a-zA-Z0-9_\-\.]{20,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----", re.IGNORECASE),
]

_CREDENTIAL_PATTERNS = [
    re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{15,}|ghp_[a-zA-Z0-9]{20,}|gho_[a-zA-Z0-9]{20,})\b"),
    _JWT_PATTERN,
    _AWS_KEY_PATTERN,
    re.compile(r"\bbearer\s+[a-zA-Z0-9_\-\.]{20,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----", re.IGNORECASE),
]

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_CANDIDATE_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")


def is_luhn_valid(number_str: str) -> bool:
    """Validate digits string using the standard Luhn checksum algorithm."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) not in (13, 14, 15, 16, 19):
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def normalize_memory_content(content: str) -> str:
    """Normalize memory content for exact matching: lowercase, strip punctuation, collapse whitespace."""
    if not content:
        return ""
    text = content.lower().strip()
    text = re.sub(r"[.,!?;:]+$", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _scan_sensitive_dict_or_list(data: Any, path: str = "") -> None:
    """Recursively scan metadata dictionaries and lists for credentials or sensitive identifiers."""
    if isinstance(data, dict):
        for k, v in data.items():
            k_clean = str(k).strip().lower()
            if k_clean in _FORBIDDEN_METADATA_KEYS:
                raise ValueError(
                    f"Metadata key '{k}' is prohibited as it may contain sensitive credentials."
                )
            _scan_sensitive_dict_or_list(v, path=f"{path}.{k}" if path else str(k))
    elif isinstance(data, (list, tuple, set)):
        for idx, item in enumerate(data):
            _scan_sensitive_dict_or_list(item, path=f"{path}[{idx}]")
    elif isinstance(data, str):
        val_lower = data.strip().lower()
        for prefix in _SENSITIVE_VALUE_PREFIXES:
            if val_lower.startswith(prefix):
                raise ValueError(
                    f"Metadata value at '{path}' contains sensitive credential patterns."
                )
        if _JWT_PATTERN.search(data):
            raise ValueError(f"Metadata value at '{path}' contains a JWT credential pattern.")
        if _AWS_KEY_PATTERN.search(data):
            raise ValueError(f"Metadata value at '{path}' contains an access key pattern.")
        for pattern in _INDIAN_IDENTIFIER_PATTERNS:
            if pattern.search(data):
                raise ValueError(
                    f"Metadata value at '{path}' contains a sensitive personal identifier."
                )


def validate_memory_content_security(content: str) -> str:
    """Validate that memory content does not contain prompt overrides, sensitive credentials, or personal identifiers."""
    cleaned = content.strip()
    if not cleaned:
        raise ValueError("Memory content cannot be empty or whitespace.")

    # 1. Check prompt injection / system override patterns
    for pattern in _SECURITY_OVERRIDE_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains prohibited system-prompt overrides or attempts to bypass security boundaries."
            )

    # 2. Check passwords and PINs
    for pattern in _PASSWORD_PIN_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive personal credentials (password or PIN)."
            )

    # 3. Check API keys, tokens, and private keys
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive credentials, API keys, or private keys."
            )

    # 4. Check government and national identifiers (Aadhaar, PAN, SSN)
    for pattern in _INDIAN_IDENTIFIER_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive personal identity numbers (Aadhaar or PAN)."
            )
    if _SSN_PATTERN.search(cleaned):
        raise ValueError("Memory content contains sensitive government identification numbers.")

    # 5. Check financial credit card details with Luhn validation
    for match in _CREDIT_CARD_CANDIDATE_PATTERN.finditer(cleaned):
        card_str = match.group()
        if is_luhn_valid(card_str):
            raise ValueError("Memory content contains sensitive financial card details.")

    # 2. Check passwords and PINs
    for pattern in _PASSWORD_PIN_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive personal credentials (password or PIN)."
            )

    # 3. Check API keys, tokens, and private keys
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive credentials, API keys, or private keys."
            )

    # 4. Check government and national identifiers (Aadhaar, PAN, SSN)
    for pattern in _INDIAN_IDENTIFIER_PATTERNS:
        if pattern.search(cleaned):
            raise ValueError(
                "Memory content contains sensitive personal identity numbers (Aadhaar or PAN)."
            )
    if _SSN_PATTERN.search(cleaned):
        raise ValueError("Memory content contains sensitive government identification numbers.")

    # 5. Check financial credit card details with Luhn validation
    for match in _CREDIT_CARD_CANDIDATE_PATTERN.finditer(cleaned):
        card_str = match.group()
        if is_luhn_valid(card_str):
            raise ValueError("Memory content contains sensitive financial card details.")

    return cleaned


def validate_memory_metadata_security(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Ensure metadata does not contain sensitive credential keys or secret values."""
    if not metadata:
        return metadata
    _scan_sensitive_dict_or_list(metadata)
    return metadata


class MessageBase(BaseModel):
    """Base attributes for conversation message data."""

    role: MessageRole = Field(description="Role of the sender: user, assistant, or system")
    content: str = Field(description="Text body of the message")
    model: str | None = Field(default=None, description="Model used for generation if applicable")
    provider: str | None = Field(default=None, description="AI provider if applicable")
    token_usage: dict[str, Any] | None = Field(
        default=None, description="Reported token usage metrics"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure message content is not empty."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty or whitespace.")
        return cleaned


class MessageCreate(MessageBase):
    """Schema for persisting a new message."""


class MessageRead(MessageBase):
    """Schema representing a stored conversation message."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique message identifier")
    conversation_id: str = Field(description="Parent conversation identifier")
    created_at: datetime = Field(description="Timestamp when message was stored")


class ConversationBase(BaseModel):
    """Base attributes for conversations."""

    title: str | None = Field(
        default=None,
        max_length=255,
        description="Optional title for the conversation",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional arbitrary metadata dictionary",
    )


class ConversationCreate(ConversationBase):
    """Schema for creating a new conversation."""


class ConversationUpdate(BaseModel):
    """Schema for updating an existing conversation."""

    title: str | None = Field(default=None, max_length=255)
    metadata: dict[str, Any] | None = Field(default=None)


class ConversationRead(BaseModel):
    """Summary schema for a conversation thread."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique conversation identifier")
    title: str = Field(description="Conversation title")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    metadata_json: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")
    message_count: int = Field(default=0, description="Total number of stored messages")


class ConversationDetail(ConversationRead):
    """Detailed conversation schema including full messages list."""

    messages: list[MessageRead] = Field(
        default_factory=list,
        description="Chronologically ordered messages in this conversation",
    )


class ConversationListResponse(BaseModel):
    """Paginated conversation list response."""

    conversations: list[ConversationRead] = Field(description="List of conversation summaries")
    total: int = Field(description="Total count of conversations")
    limit: int = Field(description="Page size limit applied")
    offset: int = Field(description="Page offset applied")


MemoryCategory = Literal["preference", "fact", "project", "biography", "instruction"]
MemoryOrigin = Literal["explicit", "inferred"]
MemoryStatus = Literal["candidate", "active", "superseded", "deleted"]


class MemoryCreate(BaseModel):
    """Schema for creating a new memory record."""

    content: str = Field(
        min_length=1,
        max_length=5000,
        description="The statement or fact to store in long-term memory",
    )
    category: MemoryCategory = Field(
        default="fact",
        description="Category classification for the memory",
    )
    origin: MemoryOrigin = Field(
        default="explicit",
        description="Origin of the memory: explicit or inferred",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    importance: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Importance score from 1.0 to 5.0",
    )
    status: MemoryStatus = Field(
        default="active",
        description="Lifecycle state for the memory",
    )
    source_conversation_id: str | None = Field(
        default=None,
        description="Optional ID of originating conversation",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata dictionary (cannot contain secrets)",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        return validate_memory_content_security(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        return validate_memory_metadata_security(v)

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Validate invariant rules for memory creation."""
        if self.status in ("superseded", "deleted"):
            raise ValueError(f"A new memory cannot be created directly in '{self.status}' state.")
        if self.origin == "inferred" and self.confidence < 0.75 and self.status == "active":
            raise ValueError(
                "An inferred memory with confidence below 0.75 cannot be created directly as active (must be 'candidate')."
            )
        if self.origin == "explicit" and self.confidence < 0.90:
            raise ValueError(
                "Explicit memories must have a high confidence score (>= 0.90, default 1.0)."
            )
        return self


class MemoryUpdate(BaseModel):
    """Schema for updating an existing memory record."""

    content: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
        description="Updated content statement",
    )
    category: MemoryCategory | None = Field(
        default=None,
        description="Updated category classification",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated confidence score",
    )
    importance: float | None = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Updated importance score",
    )
    status: MemoryStatus | None = Field(
        default=None,
        description="Updated lifecycle status",
    )
    superseded_by_id: str | None = Field(
        default=None,
        description="UUID of the newer memory record replacing this one",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Updated metadata dictionary (cannot contain secrets)",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_memory_content_security(v)
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        return validate_memory_metadata_security(v)

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        """Validate invariant rules for memory updates."""
        if self.status == "superseded" and self.superseded_by_id is None:
            raise ValueError(
                "A memory cannot enter the superseded state without a valid superseded_by_id relationship."
            )
        return self


class MemoryRead(BaseModel):
    """Schema representing a stored memory record."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique memory identifier")
    content: str = Field(description="Content of the memory")
    category: str = Field(description="Category classification")
    origin: str = Field(description="Origin: explicit or inferred")
    confidence: float = Field(description="Confidence score")
    importance: float = Field(description="Importance score")
    status: str = Field(description="Lifecycle status: candidate, active, superseded, deleted")
    superseded_by_id: str | None = Field(
        default=None,
        description="Identifier of newer memory replacing this record",
    )
    source_conversation_id: str | None = Field(
        default=None,
        description="Identifier of originating conversation",
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    last_accessed_at: datetime | None = Field(
        default=None,
        description="Last retrieval timestamp for context injection",
    )
    access_count: int = Field(
        default=0,
        description="Number of times retrieved for context injection",
    )
    metadata_json: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary",
    )


class MemoryListResponse(BaseModel):
    """Paginated response containing list of memories."""

    memories: list[MemoryRead] = Field(description="List of memory items")
    total: int = Field(description="Total count of memories matching filter")
    limit: int = Field(description="Page size limit applied")
    offset: int = Field(description="Page offset applied")


class MemoryDeleteResponse(BaseModel):
    """Response confirming soft or hard deletion of a memory record."""

    id: str = Field(description="Identifier of deleted memory")
    deleted: bool = Field(default=True, description="Whether deletion was successful")
    hard: bool = Field(description="Whether hard deletion was performed")
    message: str = Field(description="Human-readable status summary")
