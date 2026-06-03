"""Question-answering route."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends

from agent.healthcare_agent import HealthcareAgent
from agent.router import IntentRouter
from agent.tools.appointment_tool import AppointmentTool
from agent.tools.health_tool import HealthTool
from agent.tools.rag_tool import RAGTool
from core.config.settings import Settings, get_settings
from rag.pipeline import RAGPipeline
from schemas import AskRequest, AskResponse

router = APIRouter(tags=["ask"])


@lru_cache(maxsize=1)
def get_healthcare_agent() -> HealthcareAgent:
    """Build and cache the healthcare agent."""

    settings = get_settings()
    rag_pipeline = RAGPipeline.from_settings(settings)
    return HealthcareAgent(
        router=IntentRouter(),
        rag_tool=RAGTool(rag_pipeline),
        appointment_tool=AppointmentTool(settings),
        health_tool=HealthTool(settings),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    agent: HealthcareAgent = Depends(get_healthcare_agent),
    _settings: Settings = Depends(get_settings),
) -> AskResponse:
    """Route and answer a healthcare assistant question."""

    return await agent.ask(request.question)
