"""Health check API routes."""

from fastapi import APIRouter

from app.api.deps import ProviderDep
from app.core.config import get_settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(provider: ProviderDep) -> HealthResponse:
    """Check application health and provider availability."""
    settings = get_settings()
    provider_health = await provider.health_check()
    overall_status = "ok" if provider_health.healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        app=settings.app_name,
        environment=settings.app_env,
        provider=provider_health.model_dump(),
    )
