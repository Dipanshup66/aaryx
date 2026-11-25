"""AI Brain orchestrator coordinating context assembly, system persona, and LLM providers."""

import re
import time
from collections.abc import AsyncIterator, Sequence

from app.brain.prompts import format_memory_context, get_system_prompt
from app.brain.schemas import BrainChatRequest, BrainChatResponse, BrainStreamChunk
from app.core.logging import get_logger
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatStreamChunk,
    LLMProvider,
)

logger = get_logger("brain.orchestrator")

_THINKING_TAG_REGEX = re.compile(
    r"<(?:think|thought)>[\s\S]*?</(?:think|thought)>", flags=re.IGNORECASE
)
_UNCLOSED_THINKING_REGEX = re.compile(r"<(?:think|thought)>[\s\S]*?$", flags=re.IGNORECASE)


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since ``started`` (a ``time.perf_counter()`` reading)."""
    return max(0, round((time.perf_counter() - started) * 1000))


def sanitize_reply_content(content: str) -> str:
    """Strip raw internal reasoning tags (<think>...</think>) from user-facing assistant content."""
    if not content:
        return ""
    cleaned = _THINKING_TAG_REGEX.sub("", content)
    cleaned = _UNCLOSED_THINKING_REGEX.sub("", cleaned)
    return cleaned.strip()


def merge_unanswered_turns(
    history: list[ChatMessage], message: str
) -> tuple[list[ChatMessage], str]:
    """Fold unanswered user messages in *stored* history into the next prompt.

    A turn that failed after the user message was persisted (provider error, timeout,
    interrupted stream, empty reply) leaves a user message with no assistant reply. Sending it
    followed by the next user message would give the provider consecutive same-role messages,
    which Anthropic and Gemini reject. Adjacent same-role messages inside ``history`` are merged,
    and trailing user messages are folded into ``message``, so the user's text is kept and roles
    alternate. System messages are never merged. Inputs are not mutated.

    Returns:
        ``(history, message)`` safe to hand to :class:`BrainChatRequest`.
    """
    merged: list[ChatMessage] = []
    for item in history:
        if merged and item.role != "system" and merged[-1].role == item.role:
            merged[-1] = ChatMessage(
                role=item.role, content=f"{merged[-1].content}\n\n{item.content}"
            )
        else:
            merged.append(item)

    trailing: list[str] = []
    while merged and merged[-1].role == "user":
        trailing.append(merged.pop().content)
    if trailing:
        message = "\n\n".join([*reversed(trailing), message])
    return merged, message


class ThinkingStreamFilter:
    """Incrementally strips ``<think>``/``<thought>`` reasoning blocks from streamed text.

    Handles open and close tags that arrive in the same chunk, tags split across chunk
    boundaries, several blocks per response, upper/lower-case tags, and unclosed blocks
    (everything after an unclosed opener is treated as reasoning and withheld). Text that
    merely *looks like* the start of a tag at a chunk boundary is held back until the next
    chunk decides what it is, and released by :meth:`flush` if the stream ends first.

    The visible output is consistent with :func:`sanitize_reply_content` applied to the
    complete raw text (up to surrounding whitespace).
    """

    _OPEN_TAGS = ("<think>", "<thought>")
    _CLOSE_TAGS = ("</think>", "</thought>")

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False
        self.saw_thinking = False

    @property
    def in_thinking(self) -> bool:
        """True while inside a reasoning block."""
        return self._in_thinking

    def feed(self, text: str) -> str:
        """Consume a chunk of raw model text and return the user-visible portion."""
        self._buffer += text
        visible: list[str] = []

        while True:
            tags = self._CLOSE_TAGS if self._in_thinking else self._OPEN_TAGS
            lowered = self._buffer.lower()

            match_index = -1
            match_len = 0
            for tag in tags:
                idx = lowered.find(tag)
                if idx != -1 and (match_index == -1 or idx < match_index):
                    match_index, match_len = idx, len(tag)

            if match_index != -1:
                if not self._in_thinking:
                    visible.append(self._buffer[:match_index])
                    self.saw_thinking = True
                self._in_thinking = not self._in_thinking
                self._buffer = self._buffer[match_index + match_len :]
                continue

            hold = self._partial_tag_suffix_len(lowered, tags)
            cut = len(self._buffer) - hold
            if not self._in_thinking:
                visible.append(self._buffer[:cut])
            self._buffer = self._buffer[cut:]
            break

        return "".join(visible)

    def flush(self) -> str:
        """Signal end of stream: release held text unless it belongs to unclosed reasoning."""
        tail = "" if self._in_thinking else self._buffer
        self._buffer = ""
        return tail

    @staticmethod
    def _partial_tag_suffix_len(lowered: str, tags: tuple[str, ...]) -> int:
        """Length of the longest suffix of ``lowered`` that is a proper prefix of any tag."""
        longest = 0
        for tag in tags:
            for size in range(min(len(tag) - 1, len(lowered)), longest, -1):
                if lowered.endswith(tag[:size]):
                    longest = size
                    break
        return longest


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

    def _assemble_messages(
        self, request: BrainChatRequest, memories: Sequence[str] | None = None
    ) -> list[ChatMessage]:
        """Assemble system prompt, bounded conversation history, and the latest user message.

        Preserves chronological order:
        [System Persona (+ approved memories)] -> [Recent History Window] -> [Current User Message]
        """
        system_prompt = get_system_prompt() + format_memory_context(memories or [])
        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]

        # Enforce history limit by selecting the most recent N messages
        trimmed_history = (
            request.history[-self.max_history_messages :]
            if len(request.history) > self.max_history_messages
            else request.history
        )

        messages.extend(trimmed_history)
        messages.append(ChatMessage(role="user", content=request.message))
        return messages

    async def process(
        self, request: BrainChatRequest, memories: Sequence[str] | None = None
    ) -> BrainChatResponse:
        """Process a conversational request and generate a complete reply.

        ``memories`` are approved long-term memories to include in the system prompt. They are
        passed by the server, never taken from the client request.

        Ensures raw reasoning is not leaked in the reply or thinking field.
        """
        prompt_messages = self._assemble_messages(request, memories)
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

        started = time.perf_counter()
        response = await self.provider.generate(chat_request)
        latency_ms = _elapsed_ms(started)
        logger.info(
            "Reply ready (provider=%s, model=%s, latency_ms=%d)",
            response.provider or "unknown",
            response.model,
            latency_ms,
        )

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
            latency_ms=latency_ms,
        )

    async def process_stream(
        self, request: BrainChatRequest, memories: Sequence[str] | None = None
    ) -> AsyncIterator[BrainStreamChunk]:
        """Process a conversational request and stream reply tokens in real-time.

        Emits a sanitized indicator ('AARYX is thinking...') when thinking is active,
        suppressing raw chain-of-thought tokens from streaming to the client.
        """
        prompt_messages = self._assemble_messages(request, memories)
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
        tag_filter = ThinkingStreamFilter()
        last_chunk: ChatStreamChunk | None = None
        started = time.perf_counter()

        def indicator(source: ChatStreamChunk) -> BrainStreamChunk:
            return BrainStreamChunk(
                delta="",
                model=source.model,
                finish_reason=None,
                usage=None,
                thinking="AARYX is thinking...",
                provider=source.provider,
            )

        async for chunk in self.provider.stream(chat_request):
            last_chunk = chunk

            # 1. Native provider thinking (e.g., Ollama message.thinking): never forwarded.
            #    Any answer text carried by the same chunk is still processed below.
            if chunk.thinking and not thinking_indicator_emitted:
                thinking_indicator_emitted = True
                yield indicator(chunk)

            # 2. Strip embedded <think>/<thought> blocks, tolerating split and repeated tags.
            delta = tag_filter.feed(chunk.delta or "")
            if tag_filter.saw_thinking and not thinking_indicator_emitted:
                thinking_indicator_emitted = True
                yield indicator(chunk)

            # Nothing more can follow a finish signal, so release any held-back text now.
            if chunk.finish_reason:
                delta += tag_filter.flush()

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
                latency_ms=_elapsed_ms(started) if chunk.finish_reason else None,
            )

        # Stream ended without a finish signal: release text that was held back.
        tail = tag_filter.flush()
        if tail and last_chunk is not None:
            yield BrainStreamChunk(
                delta=tail,
                model=last_chunk.model,
                finish_reason=None,
                usage=None,
                thinking=None,
                provider=last_chunk.provider,
                latency_ms=_elapsed_ms(started),
            )
