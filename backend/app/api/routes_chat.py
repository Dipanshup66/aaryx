"""Conversational chat routes integrating AI Brain orchestration and persistent memory."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import MemoryDep, OrchestratorDep
from app.brain.orchestrator import sanitize_reply_content
from app.brain.schemas import BrainChatRequest, BrainChatResponse
from app.core.logging import get_logger
from app.memory.exceptions import ConversationNotFoundError

logger = get_logger("api.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=BrainChatResponse)
async def chat(
    request: BrainChatRequest,
    orchestrator: OrchestratorDep,
    memory_service: MemoryDep,
) -> BrainChatResponse:
    """Send a conversational message to AARYX and receive a complete response.

    If memory is enabled, automatically manages conversation persistence and context retrieval.
    """
    conversation_id = request.conversation_id
    effective_history = list(request.history)

    if memory_service is not None:
        try:
            conversation_id, stored_history = await memory_service.prepare_turn(
                conversation_id=conversation_id,
                user_prompt=request.message,
            )
            if stored_history:
                effective_history = stored_history
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            ) from exc

    orchestrator_request = BrainChatRequest(
        conversation_id=conversation_id,
        message=request.message,
        history=effective_history,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    response = await orchestrator.process(orchestrator_request)

    # Persist assistant response after successful completion
    if memory_service is not None and conversation_id:
        await memory_service.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.reply,
            model=response.model,
            provider=response.provider,
            token_usage=response.usage.model_dump() if response.usage else None,
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

    if memory_service is not None:
        try:
            conversation_id, stored_history = await memory_service.prepare_turn(
                conversation_id=conversation_id,
                user_prompt=request.message,
            )
            if stored_history:
                effective_history = stored_history
        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            ) from exc

    orchestrator_request = BrainChatRequest(
        conversation_id=conversation_id,
        message=request.message,
        history=effective_history,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    async def event_generator() -> AsyncIterator[str]:
        accumulated_tokens: list[str] = []
        last_model: str | None = None
        last_provider: str | None = None
        last_usage = None
        completed_successfully = False

        try:
            async for chunk in orchestrator.process_stream(orchestrator_request):
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

            completed_successfully = True
            yield "data: [DONE]\n\n"
        finally:
            if completed_successfully and accumulated_tokens and memory_service and conversation_id:
                full_reply = sanitize_reply_content("".join(accumulated_tokens))
                await memory_service.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_reply,
                    model=last_model,
                    provider=last_provider,
                    token_usage=last_usage.model_dump() if last_usage else None,
                )
            elif not completed_successfully and conversation_id:
                logger.warning(
                    "Streaming terminated without completion for conversation %s; "
                    "partial assistant response will not be stored.",
                    conversation_id,
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
