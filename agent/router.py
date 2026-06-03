"""Intent router for the single healthcare agent."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentRoute(StrEnum):
    """Supported single-agent routes."""

    RAG = "rag"
    APPOINTMENT = "appointment_tool"
    HEALTH = "health_tool"


@dataclass(frozen=True)
class IntentDecision:
    """Intent routing decision."""

    route: AgentRoute
    reason: str


class IntentRouter:
    """Route user requests to a tool or the RAG pipeline."""

    appointment_keywords = {
        "appointment",
        "book",
        "booking",
        "schedule",
        "availability",
        "available",
        "visit",
    }
    health_keywords = {"health check", "status", "system health"}

    def route(self, query: str) -> IntentDecision:
        """Detect the best route for a query."""

        normalized = query.lower()
        if any(keyword in normalized for keyword in self.health_keywords):
            return IntentDecision(AgentRoute.HEALTH, "System health intent detected.")
        if any(keyword in normalized for keyword in self.appointment_keywords):
            return IntentDecision(
                AgentRoute.APPOINTMENT,
                "Appointment scheduling intent detected.",
            )
        return IntentDecision(AgentRoute.RAG, "Knowledge-base question detected.")
