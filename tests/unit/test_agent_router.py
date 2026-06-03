"""Unit tests for agent routing."""

from __future__ import annotations

from agent.router import AgentRoute, IntentRouter


def test_router_detects_appointment_intent() -> None:
    """Appointment requests should route to the appointment tool."""

    decision = IntentRouter().route("Can I book an appointment tomorrow?")

    assert decision.route == AgentRoute.APPOINTMENT


def test_router_defaults_to_rag() -> None:
    """Knowledge questions should route to RAG."""

    decision = IntentRouter().route("What does the telehealth policy say?")

    assert decision.route == AgentRoute.RAG
