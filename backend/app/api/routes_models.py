"""Model discovery and inspection routes."""

from fastapi import APIRouter

from app.api.deps import ProviderDep
from app.providers.base import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelInfo])
async def list_models(provider: ProviderDep) -> list[ModelInfo]:
    """List available local or cloud models on the active provider."""
    return await provider.list_models()
