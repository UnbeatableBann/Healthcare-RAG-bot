"""Health tool for agent-routed system health requests."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from schemas import AskResponse


class HealthTool:
    """Return basic application health information."""

    name = "health_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def run(self, _query: str) -> AskResponse:
        """Return health information."""

        answer = (
            f"{self.settings.APP_NAME} is running in "
            f"{self.settings.ENVIRONMENT} mode."
        )
        return AskResponse(
            answer=answer,
            citations=[],
            confidence_score=1.0,
            route=self.name,
            answerable=True,
        )
