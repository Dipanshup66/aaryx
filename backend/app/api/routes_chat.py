"""Conversational chat routes integrating AI Brain orchestration and persistent memory."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.api.deps import MemoryDep, OrchestratorDep
from app.brain.orchestrator import merge_unanswered_turns, sanitize_reply_content
from app.brain.schemas import BrainChatRequest, BrainChatResponse
from app.core.logging import get_logger
from app.memory.exceptions import ConversationNotFoundError
from app.memory.service import MemoryService
from app.providers.base import ProviderError, error_payload

logger = get_logger("api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


def _log_failed_turn(conversation_id: str | None, exc: Exception) -> None:
    """Record a failed turn without logging message content or raw provider payloads."""
    logger.warning(
        "Chat turn failed for conversation %s (%s); user message retained, "
        "no assistant message stored.",
        conversation_id,
        exc.__class__.__name__,
    )


async def _load_memory_context(memory_service: MemoryService | None, message: str) -> list[str]:
    """Fetch approved memories for the prompt. A memory problem never breaks the chat turn."""
    if memory_service is None:
        return []
    try:
        memories = await memory_service.get_context_memories(message)
    except Exception as exc:  # noqa: BLE001 - memory must never break a chat turn
        logger.warning(
            "Could not load memory context (%s); continuing without it.", exc.__class__.__name__
        )
        return []
    return [m.content for m in memories]


async def _suggest_memories(
    memory_service: MemoryService,
    user_prompt: str,
    assistant_reply: str,
    conversation_id: str,
) -> None:
    """Run after the reply is delivered: save "remember this" requests and queue suggestions.

    Inferred memories are stored only as pending suggestions (Decision D-001). Any failure is
    logged without message content and never affects the chat turn.
    """
    try:
        stored = await memory_service.extract_and_store_memories(
            user_prompt=user_prompt,
            assistant_reply=assistant_reply,
            conversation_id=conversation_id,
        )
        if stored:
            logger.info(
                "Memory update after turn in conversation %s: %d stored",
                conversation_id,
                len(stored),
            )
    except Exception as exc:  # noqa: BLE001 - background step must never fail the turn
        logger.warning(
            "Memory extraction failed for conversation %s (%s); chat turn unaffected.",
            conversation_id,
            exc.__class__.__name__,
        )


@router.post("", response_model=BrainChatResponse)
async def chat(
    request: BrainChatRequest,
    orchestrator: OrchestratorDep,
    memory_service: MemoryDep,
    background_tasks: BackgroundTasks,
) -> BrainChatResponse:
    """Send a conversational message to AARYX and receive a complete response.

    If memory is enabled, automatically manages conversation persistence and context retrieval.
    """
    conversation_id = request.conversation_id
    effective_history = list(request.history)
    effective_message = request.message

    if memory_service is not None:
        try:
            conversation_id, stored_history = await memory_service.prepare_turn(
                conversation_id=conversation_id,
                user_prompt=request.message,
            )
            if stored_history:
                # Heal unanswered user messages left behind by earlier failed turns.
                effective_history, effective_message = merge_unanswered_turns(
                    stored_history, request.message
                )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            ) from exc

    orchestrator_request = BrainChatRequest(
        conversation_id=conversation_id,
        message=effective_message,
        history=effective_history,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    memories = await _load_memory_context(memory_service, request.message)
    process_kwargs = {"memories": memories} if memories else {}

    try:
        response = await orchestrator.process(orchestrator_request, **process_kwargs)
    except Exception as exc:
        # The user message was persisted before inference and is intentionally retained.
        # No assistant message is stored; the next turn's prompt merges the unanswered message.
        _log_failed_turn(conversation_id, exc)
        raise

    # Persist assistant response after successful completion. A generation that produced no
    # visible text (empty or reasoning-only output) is a valid, completed generation: there is
    # simply nothing to store, and that must not turn into an HTTP error.
    if memory_service is not None and conversation_id:
        if response.reply.strip():
            await memory_service.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response.reply,
                model=response.model,
                provider=response.provider,
                token_usage=response.usage.model_dump() if response.usage else None,
            )
            background_tasks.add_task(
                _suggest_memories,
                memory_service,
                request.message,
                response.reply,
                conversation_id,
            )
        else:
            logger.warning(
                "Model returned no visible reply for conversation %s "
                "(empty or reasoning-only output); no assistant message stored.",
                conversation_id,
            )

    return response.model_copy(update={"conversation_id": conversation_id})


@router.post("/stream")
async def chat_stream(
    request: BrainChatRequest,
    orchestrator: OrchestratorDep,
    memory_service: MemoryDep,
) -> StreamingResponse:
    """Stream response tokens in real-time using Server-Sent Events (SSE).

    Persists user message prior to stream execution and assistant reply upon successful completion.
    """
    conversation_id = request.conversation_id
    effective_history = list(request.history)
    effective_message = request.message

    if memory_service is not None:
        try:
            conversation_id, stored_history = await memory_service.prepare_turn(
                conversation_id=conversation_id,
                user_prompt=request.message,
            )
            if stored_history:
                # Heal unanswered user messages left behind by earlier failed turns.
                effective_history, effective_message = merge_unanswered_turns(
                    stored_history, request.message
                )
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            ) from exc

    orchestrator_request = BrainChatRequest(
        conversation_id=conversation_id,
        message=effective_message,
        history=effective_history,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    memories = await _load_memory_context(memory_service, request.message)
    process_kwargs = {"memories": memories} if memories else {}
    completed_reply: list[str] = []

    async def event_generator() -> AsyncIterator[str]:
        accumulated_tokens: list[str] = []
        last_model: str | None = None
        last_provider: str | None = None
        last_usage = None
        completed = False
        failure: str | None = None

        try:
            async for chunk in orchestrator.process_stream(orchestrator_request, **process_kwargs):
                enriched_chunk = chunk.model_copy(update={"conversation_id": conversation_id})
                yield f"data: {enriched_chunk.model_dump_json()}\n\n"

                if chunk.delta:
                    accumulated_tokens.append(chunk.delta)
                if chunk.model:
                    last_model = chunk.model
                if chunk.provider:
                    last_provider = chunk.provider
                if chunk.usage:
                    last_usage = chunk.usage

            # Persist BEFORE signalling completion so [DONE] is never sent for a turn whose
            # persistence failed. A reply with no visible text has nothing to store.
            if memory_service and conversation_id:
                full_reply = sanitize_reply_content("".join(accumulated_tokens))
                if full_reply:
                    await memory_service.save_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_reply,
                        model=last_model,
                        provider=last_provider,
                        token_usage=last_usage.model_dump() if last_usage else None,
                    )
                    completed_reply.append(full_reply)
                else:
                    logger.warning(
                        "Model returned no visible reply for conversation %s "
                        "(empty or reasoning-only output); no assistant message stored.",
                        conversation_id,
                    )

            completed = True
            yield "data: [DONE]\n\n"
        except ProviderError as exc:
            # The status line is already sent, so tell the client why the stream stops: the same
            # fields (code, detail, hint, model) as an error response, in an SSE event with an
            # "error" key. A failed stream never ends with [DONE].
            failure = exc.__class__.__name__
            yield f"data: {json.dumps({'error': error_payload(exc)})}\n\n"
            raise
        except Exception as exc:
            failure = exc.__class__.__name__
            raise
        finally:
            if not completed:
                logger.warning(
                    "Streaming turn for conversation %s ended without completion (%s); "
                    "user message retained, partial assistant response not stored.",
                    conversation_id,
                    failure or "client disconnected or cancelled",
                )

    async def after_stream() -> None:
        # Runs only for a turn that completed and was persisted.
        if memory_service and conversation_id and completed_reply:
            await _suggest_memories(
                memory_service, request.message, completed_reply[0], conversation_id
            )

    return StreamingResponse(
        event_generator(),
        background=BackgroundTask(after_stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
