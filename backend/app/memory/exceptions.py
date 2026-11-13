"""Domain exceptions for conversation memory and persistence."""


class MemoryError(Exception):
    """Base exception for persistent memory errors."""


class ConversationNotFoundError(MemoryError):
    """Raised when a referenced conversation is not found in storage."""

    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        super().__init__(f"Conversation '{conversation_id}' not found.")


class ConversationValidationError(MemoryError):
    """Raised when conversation or message data fails validation."""


class MemoryNotFoundError(MemoryError):
    """Raised when a referenced memory entity is not found in storage."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(f"Memory '{memory_id}' not found.")


class MemoryValidationError(MemoryError):
    """Raised when memory entity data fails validation or violates security rules."""
