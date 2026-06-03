"""Health check route for the API foundation."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from core.config.settings import Settings, get_settings
from schemas.base import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return service health and configuration visibility."""

    return HealthResponse(
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
        status="ok",
        environment=settings.ENVIRONMENT,
        metrics_enabled=settings.METRICS_ENABLED,
    )

