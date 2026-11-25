"""Data schemas for AI Brain orchestration and conversation exchange."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.providers.base import ChatMessage, TokenUsage

MessageRole = Literal["system", "user", "assistant"]


class BrainChatRequest(BaseModel):
    """Incoming user request for the AI Brain."""

    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation identifier for persistent memory",
    )
    message: str = Field(description="Current user prompt or message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation history in chronological order",
    )
    model: str | None = Field(default=None, description="Optional override model name")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Optional sampling temperature override"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Optional maximum tokens to generate"
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        """Ensure the user message is non-empty and not only whitespace."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("User message cannot be empty or whitespace.")
        return cleaned


class BrainChatResponse(BaseModel):
    """Normalized response produced by the AI Brain."""

    reply: str = Field(description="AI response content")
    model: str = Field(description="Model used for generation")
    finish_reason: str | None = Field(default="stop", description="Generation completion reason")
    usage: TokenUsage | None = Field(default=None, description="Token usage metrics if available")
    thinking: str | None = Field(default=None, description="Thinking trace if reported")
    provider: str | None = Field(default=None, description="Provider that generated the response")
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Milliseconds from sending the request to the model until the reply was ready "
        "(includes retries and fallback)",
    )
    conversation_id: str | None = Field(default=None, description="Active conversation identifier")


class BrainStreamChunk(BaseModel):
    """Incremental response token yielded during streaming."""

    delta: str = Field(description="Incremental text content")
    model: str | None = Field(default=None, description="Model generating the stream")
    finish_reason: str | None = Field(default=None, description="Completion status on final chunk")
    usage: TokenUsage | None = Field(default=None, description="Token usage metrics on final chunk")
    thinking: str | None = Field(default=None, description="Thinking chunk if reported")
    provider: str | None = Field(default=None, description="Provider generating this stream chunk")
    latency_ms: int | None = Field(
        default=None,
        ge=0,
        description="Total milliseconds the model took; set only on the last chunk of a stream",
    )
    conversation_id: str | None = Field(default=None, description="Active conversation identifier")
