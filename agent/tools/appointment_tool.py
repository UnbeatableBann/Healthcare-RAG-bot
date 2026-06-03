"""Appointment tool for operational appointment requests."""

from __future__ import annotations

from core.config.settings import Settings, get_settings
from schemas import AskResponse


class AppointmentTool:
    """Handle appointment booking and availability requests."""

    name = "appointment_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def run(self, query: str) -> AskResponse:
        """Return configured appointment availability and contact details."""

        answer = (
            f"{self.settings.APPOINTMENT_CLINIC_NAME} appointment support is available "
            f"{self.settings.APPOINTMENT_HOURS}. To request scheduling help, contact "
            f"{self.settings.APPOINTMENT_CONTACT}. Request received: {query.strip()}"
        )
        return AskResponse(
            answer=answer,
            citations=[],
            confidence_score=1.0,
            route=self.name,
            answerable=True,
        )
