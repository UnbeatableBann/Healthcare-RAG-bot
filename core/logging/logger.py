"""Loguru logging configuration."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from core.config.settings import Settings, get_settings

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
    "{name}:{function}:{line} | {message}"
)

_configured = False


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure Loguru once for the process."""

    global _configured
    if _configured and not force:
        return

    resolved_settings = settings or get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=resolved_settings.LOG_LEVEL,
        format=LOG_FORMAT,
        serialize=resolved_settings.LOG_JSON,
        backtrace=resolved_settings.ENVIRONMENT == "development",
        diagnose=resolved_settings.ENVIRONMENT == "development",
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    """Return a configured Loguru logger bound to a module name."""

    if name is None:
        return logger
    return logger.bind(module=name)
