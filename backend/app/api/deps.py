"""FastAPI dependency injection utilities."""

from typing import Annotated

from fastapi import Depends, Request

from app.brain.orchestrator import BrainOrchestrator
from app.memory.service import MemoryService
from app.providers.base import LLMProvider


def get_provider(request: Request) -> LLMProvider:
    """Retrieve the active LLMProvider from application state."""
    return request.app.state.provider


def get_orchestrator(request: Request) -> BrainOrchestrator:
    """Retrieve the active BrainOrchestrator from application state."""
    return request.app.state.orchestrator


def get_memory_service(request: Request) -> MemoryService | None:
    """Retrieve the active MemoryService from application state if enabled."""
    return getattr(request.app.state, "memory_service", None)


ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
OrchestratorDep = Annotated[BrainOrchestrator, Depends(get_orchestrator)]
MemoryDep = Annotated[MemoryService | None, Depends(get_memory_service)]
