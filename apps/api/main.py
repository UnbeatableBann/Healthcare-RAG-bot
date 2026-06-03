"""FastAPI application factory for the Healthcare AI Assistant."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.routes import ask, health, ingest
from core.config.settings import Settings, get_settings
from core.exceptions import HealthcareAssistantError
from core.logging.logger import configure_logging, get_logger
from core.observability.metrics import (
    API_LATENCY_SECONDS,
    API_REQUESTS_TOTAL,
    CONTENT_TYPE_LATEST,
    render_prometheus_metrics,
)
from schemas.base import ErrorDetail, ErrorResponse

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings, force=True)

    app = FastAPI(
        title=resolved_settings.APP_NAME,
        version=resolved_settings.APP_VERSION,
        description=(
            "Production-oriented agentic hybrid RAG platform for healthcare "
            "knowledge management."
        ),
        docs_url=f"{resolved_settings.API_PREFIX}/docs",
        redoc_url=f"{resolved_settings.API_PREFIX}/redoc",
        openapi_url=f"{resolved_settings.API_PREFIX}/openapi.json",
    )

    app.state.settings = resolved_settings

    register_middlewares(app, resolved_settings)
    register_exception_handlers(app)
    register_routes(app, resolved_settings)

    logger.info(
        "FastAPI application configured",
        app_name=resolved_settings.APP_NAME,
        environment=resolved_settings.ENVIRONMENT,
        api_prefix=resolved_settings.API_PREFIX,
    )
    return app


def register_routes(app: FastAPI, settings: Settings) -> None:
    """Register application routes."""

    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(ask.router, prefix=settings.API_PREFIX)
    app.include_router(ingest.router, prefix=settings.API_PREFIX)

    if settings.METRICS_ENABLED:

        @app.get("/metrics", include_in_schema=False)
        def metrics() -> Response:
            """Expose Prometheus metrics."""

            return Response(
                content=render_prometheus_metrics(),
                media_type=CONTENT_TYPE_LATEST,
            )


def register_middlewares(app: FastAPI, settings: Settings) -> None:
    """Register cross-cutting HTTP middlewares."""

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if not settings.METRICS_ENABLED:
        return

    @app.middleware("http")
    async def metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Record request counts and latency for Prometheus."""

        start_time = time.perf_counter()
        status_code = 500
        route_path = request.url.path

        try:
            response = await call_next(request)
            status_code = response.status_code
            route = request.scope.get("route")
            route_path = getattr(route, "path", route_path)
            return response
        finally:
            elapsed = time.perf_counter() - start_time
            API_REQUESTS_TOTAL.labels(
                method=request.method,
                path=route_path,
                status_code=str(status_code),
            ).inc()
            API_LATENCY_SECONDS.labels(
                method=request.method,
                path=route_path,
            ).observe(elapsed)


def register_exception_handlers(app: FastAPI) -> None:
    """Register consistent JSON exception handlers."""

    @app.exception_handler(HealthcareAssistantError)
    async def healthcare_error_handler(
        _request: Request,
        exc: HealthcareAssistantError,
    ) -> JSONResponse:
        """Convert domain exceptions into API error responses."""

        logger.warning(
            "Domain error handled",
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
        )
        payload = ErrorResponse(
            error=ErrorDetail(
                code=exc.error_code,
                message=exc.message,
                details=exc.details,
            )
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Convert request validation errors into a stable response shape."""

        payload = ErrorResponse(
            error=ErrorDetail(
                code="REQUEST_VALIDATION_ERROR",
                message="The request payload failed validation.",
                details={"errors": exc.errors()},
            )
        )
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Convert unexpected errors into a stable response shape."""

        logger.exception("Unhandled API error", error=str(exc))
        payload = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected server error occurred.",
            )
        )
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))


app = create_app()


def run() -> None:
    """Run the API using Uvicorn."""

    settings = get_settings()
    uvicorn.run(
        "apps.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_config=None,
    )
