"""Single-agent healthcare workflow."""

from __future__ import annotations

from agent.router import AgentRoute, IntentRouter
from agent.tools.appointment_tool import AppointmentTool
from agent.tools.health_tool import HealthTool
from agent.tools.rag_tool import RAGTool
from core.observability.metrics import AGENT_LATENCY_SECONDS, TOOL_CALLS_TOTAL, observe_latency
from schemas import AskResponse


class HealthcareAgent:
    """Single-agent architecture that routes to tools or RAG."""

    def __init__(
        self,
        *,
        router: IntentRouter,
        rag_tool: RAGTool,
        appointment_tool: AppointmentTool,
        health_tool: HealthTool,
    ) -> None:
        self.router = router
        self.rag_tool = rag_tool
        self.appointment_tool = appointment_tool
        self.health_tool = health_tool

    async def ask(self, question: str) -> AskResponse:
        """Route and execute a user question."""

        decision = self.router.route(question)
        with observe_latency(AGENT_LATENCY_SECONDS, route=decision.route.value):
            if decision.route == AgentRoute.APPOINTMENT:
                response = await self.appointment_tool.run(question)
            elif decision.route == AgentRoute.HEALTH:
                response = await self.health_tool.run(question)
            else:
                response = await self.rag_tool.run(question)
        TOOL_CALLS_TOTAL.labels(
            tool_name=response.route,
            outcome="success" if response.answerable else "refused",
        ).inc()
        return response
