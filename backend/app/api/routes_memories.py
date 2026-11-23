"""Long-term memory management endpoints for persistent memory."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import MemoryDep
from app.memory.exceptions import MemoryNotFoundError, MemoryValidationError
from app.memory.schemas import (
    MemoryApprove,
    MemoryCategory,
    MemoryCreate,
    MemoryDeleteResponse,
    MemoryListResponse,
    MemoryRead,
    MemoryStatus,
    MemoryUpdate,
)

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post(
    "",
    response_model=MemoryRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new long-term memory",
)
async def create_memory(
    payload: MemoryCreate,
    memory_service: MemoryDep,
) -> MemoryRead:
    """Create a new long-term memory record."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        return await memory_service.create_memory(payload)
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List long-term memories",
)
async def list_memories(
    memory_service: MemoryDep,
    status_filter: Annotated[
        MemoryStatus | None,
        Query(
            alias="status",
            description="Filter by status (active, candidate = pending suggestions, rejected, superseded, deleted)",
        ),
    ] = None,
    category_filter: Annotated[
        MemoryCategory | None,
        Query(
            alias="category",
            description="Filter by category (preference, fact, project, biography, instruction)",
        ),
    ] = None,
    include_deleted: Annotated[
        bool,
        Query(description="Whether to include soft-deleted memories in results"),
    ] = False,
    limit: Annotated[int, Query(ge=1, le=100, description="Max items to return")] = 50,
    offset: Annotated[int, Query(ge=0, description="Offset for pagination")] = 0,
) -> MemoryListResponse:
    """Retrieve long-term memories with optional status and category filtering."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    return await memory_service.list_memories(
        status=status_filter,
        category=category_filter,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{memory_id}",
    response_model=MemoryRead,
    summary="Get a long-term memory by ID",
)
async def get_memory(
    memory_id: str,
    memory_service: MemoryDep,
) -> MemoryRead:
    """Retrieve a single long-term memory record by its UUID."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        mem = await memory_service.get_memory(memory_id)
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory '{memory_id}' not found.",
        )

    return mem


@router.patch(
    "/{memory_id}",
    response_model=MemoryRead,
    summary="Update a long-term memory",
)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    memory_service: MemoryDep,
) -> MemoryRead:
    """Update fields of an existing long-term memory record."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        return await memory_service.update_memory(memory_id, payload)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{memory_id}/approve",
    response_model=MemoryRead,
    summary="Approve a memory suggestion",
)
async def approve_memory(
    memory_id: str,
    memory_service: MemoryDep,
    payload: MemoryApprove | None = None,
) -> MemoryRead:
    """Approve a pending suggestion so it becomes an active memory (optionally edited)."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        return await memory_service.approve_memory(memory_id, payload)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/{memory_id}/reject",
    response_model=MemoryRead,
    summary="Reject a memory suggestion",
)
async def reject_memory(
    memory_id: str,
    memory_service: MemoryDep,
) -> MemoryRead:
    """Reject a pending suggestion. It is never used and will not be suggested again."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        return await memory_service.reject_memory(memory_id)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{memory_id}",
    response_model=MemoryDeleteResponse,
    summary="Delete a long-term memory",
)
async def delete_memory(
    memory_id: str,
    memory_service: MemoryDep,
    hard: Annotated[
        bool,
        Query(description="Set to true for permanent hard deletion; defaults to soft delete"),
    ] = False,
) -> MemoryDeleteResponse:
    """Delete a memory record (soft deletion by default; hard deletion if hard=true)."""
    if not memory_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistent conversation memory is disabled.",
        )

    try:
        return await memory_service.delete_memory(memory_id, hard=hard)
    except MemoryNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MemoryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
