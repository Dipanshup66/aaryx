"""Conversation management endpoints for persistent memory."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import MemoryDep
from app.memory.exceptions import ConversationNotFoundError, ConversationValidationError
from app.memory.schemas import (
    ConversationCreate,
    ConversationDetail,
    ConversationListResponse,
    ConversationRead,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post(
    "",
    response_model=ConversationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation",
)
async def create_conversation(
    payload: ConversationCreate,
    memory_service: MemoryDep,
) -> ConversationRead:
    """Create a new independent conversation thread."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    return await memory_service.create_conversation(
        title=payload.title,
        metadata=payload.metadata,
    )


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List conversations",
)
async def list_conversations(
    memory_service: MemoryDep,
    limit: Annotated[int, Query(ge=1, le=100, description="Max items to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Offset for pagination")] = 0,
) -> ConversationListResponse:
    """Retrieve conversations ordered by recent activity."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    return await memory_service.list_conversations(limit=limit, offset=offset)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation detail with messages",
)
async def get_conversation(
    conversation_id: str,
    memory_service: MemoryDep,
) -> ConversationDetail:
    """Retrieve full conversation details including chronological messages."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        conv = await memory_service.get_conversation(
            conversation_id=conversation_id,
            include_messages=True,
        )
    except (ConversationNotFoundError, ConversationValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if not conv or not isinstance(conv, ConversationDetail):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    return conv


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete conversation",
)
async def delete_conversation(
    conversation_id: str,
    memory_service: MemoryDep,
) -> dict[str, str]:
    """Delete a conversation and all its cascaded messages."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        deleted = await memory_service.delete_conversation(conversation_id)
    except ConversationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    return {
        "message": "Conversation deleted successfully",
        "id": conversation_id,
    }
