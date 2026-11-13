"""AI Brain orchestrator coordinating context assembly, system persona, and LLM providers."""

import re
from collections.abc import AsyncIterator

from app.brain.prompts import get_system_prompt
from app.brain.schemas import BrainChatRequest, BrainChatResponse, BrainStreamChunk
from app.core.logging import get_logger
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    LLMProvider,
)

logger = get_logger("brain.orchestrator")

_THINKING_TAG_REGEX = re.compile(
    r"<(?:think|thought)>[\s\S]*?</(?:think|thought)>", flags=re.IGNORECASE
)
_UNCLOSED_THINKING_REGEX = re.compile(r"<(?:think|thought)>[\s\S]*?$", flags=re.IGNORECASE)


def sanitize_reply_content(content: str) -> str:
    """Strip raw internal reasoning tags (<think>...</think>) from user-facing assistant content."""
    if not content:
        return ""
    cleaned = _THINKING_TAG_REGEX.sub("", content)
    cleaned = _UNCLOSED_THINKING_REGEX.sub("", cleaned)
    return cleaned.strip()


class BrainOrchestrator:
    """Coordinates AI reasoning, context assembly, and model provider execution.

    Decoupled from specific LLM implementations via the LLMProvider interface.
    Enforces privacy boundaries by preventing raw internal reasoning from leaking to clients.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_history_messages: int = 20,
        default_temperature: float = 0.7,
    ) -> None:
        """Initialize the AI Brain orchestrator.

        Args:
            provider: Injected LLM provider implementation.
            max_history_messages: Maximum prior messages retained in the prompt context.
            default_temperature: Default sampling temperature when not specified.
        """
        self.provider = provider
        self.max_history_messages = max_history_messages
        self.default_temperature = default_temperature

    def _assemble_messages(self, request: BrainChatRequest) -> list[ChatMessage]:
        """Assemble system prompt, bounded conversation history, and the latest user message.

        Preserves chronological order:
        [System Persona] -> [Recent History Window] -> [Current User Message]
        """
        messages: list[ChatMessage] = [ChatMessage(role="system", content=get_system_prompt())]

        # Enforce history limit by selecting the most recent N messages
        trimmed_history = (
            request.history[-self.max_history_messages :]
            if len(request.history) > self.max_history_messages
            else request.history
        )

        messages.extend(trimmed_history)
        messages.append(ChatMessage(role="user", content=request.message))
        return messages

    async def process(self, request: BrainChatRequest) -> BrainChatResponse:
        """Process a conversational request and generate a complete reply.

        Ensures raw reasoning is not leaked in the reply or thinking field.
        """
        prompt_messages = self._assemble_messages(request)
        temperature = (
            request.temperature if request.temperature is not None else self.default_temperature
        )

        chat_request = ChatRequest(
            messages=prompt_messages,
            model=request.model,
            temperature=temperature,
            max_tokens=request.max_tokens,
        )

        logger.info(
            "Processing user request (history_count=%d, total_messages=%d, model_override=%s)",
            len(request.history),
            len(prompt_messages),
            request.model or "default",
        )

        response = await self.provider.generate(chat_request)

        # Sanitize reply content to remove any embedded reasoning tags
        clean_reply = sanitize_reply_content(response.content)

        # Do not expose raw internal reasoning chain to clients
        return BrainChatResponse(
            reply=clean_reply,
            model=response.model,
            finish_reason=response.finish_reason,
            usage=response.usage,
            thinking=None,
            provider=response.provider,
        )

    async def process_stream(self, request: BrainChatRequest) -> AsyncIterator[BrainStreamChunk]:
        """Process a conversational request and stream reply tokens in real-time.

        Emits a sanitized indicator ('AARYX is thinking...') when thinking is active,
        suppressing raw chain-of-thought tokens from streaming to the client.
        """
        prompt_messages = self._assemble_messages(request)
        temperature = (
            request.temperature if request.temperature is not None else self.default_temperature
        )

        chat_request = ChatRequest(
            messages=prompt_messages,
            model=request.model,
            temperature=temperature,
            max_tokens=request.max_tokens,
        )

        logger.info(
            "Opening real-time streaming response (history_count=%d, total_messages=%d)",
            len(request.history),
            len(prompt_messages),
        )

        thinking_indicator_emitted = False
        in_think_tag = False

        async for chunk in self.provider.stream(chat_request):
            # 1. Handle native provider thinking (e.g., Ollama message.thinking)
            if chunk.thinking:
                if not thinking_indicator_emitted:
                    thinking_indicator_emitted = True
                    yield BrainStreamChunk(
                        delta="",
                        model=chunk.model,
                        finish_reason=None,
                        usage=None,
                        thinking="AARYX is thinking...",
                        provider=chunk.provider,
                    )
                # Suppress raw internal reasoning tokens
                continue

            # 2. Filter any embedded <think>...</think> tags inside delta stream
            delta = chunk.delta or ""
            if "<think>" in delta:
                in_think_tag = True
                if not thinking_indicator_emitted:
                    thinking_indicator_emitted = True
                    yield BrainStreamChunk(
                        delta="",
                        model=chunk.model,
                        finish_reason=None,
                        usage=None,
                        thinking="AARYX is thinking...",
                        provider=chunk.provider,
                    )
                parts = delta.split("<think>", 1)
                delta = parts[0]
            elif in_think_tag:
                if "</think>" in delta:
                    in_think_tag = False
                    parts = delta.split("</think>", 1)
                    delta = parts[1]
                else:
                    delta = ""

            # Skip empty chunks unless carrying finish_reason or usage metrics
            if not delta and not chunk.finish_reason and not chunk.usage:
                continue

            yield BrainStreamChunk(
                delta=delta,
                model=chunk.model,
                finish_reason=chunk.finish_reason,
                usage=chunk.usage,
                thinking=None,
                provider=chunk.provider,
            )
