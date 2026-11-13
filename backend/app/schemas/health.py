"""Health and status schemas."""

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(description="System operational status (ok or degraded)")
    app: str = Field(description="Application name")
    environment: str = Field(description="Active environment")
    provider: dict[str, Any] = Field(description="Active LLM provider diagnostics")
