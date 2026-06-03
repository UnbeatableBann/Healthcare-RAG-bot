"""End-to-end API tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.routes.ask import get_healthcare_agent
from schemas import AskResponse


class FakeAgent:
    """Fake agent used for API tests."""

    async def ask(self, question: str) -> AskResponse:
        return AskResponse(
            answer=f"received: {question}",
            citations=[],
            confidence_score=1.0,
            route="rag",
            answerable=True,
        )


def test_health_endpoint() -> None:
    """Health endpoint should return service status."""

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_endpoint_with_dependency_override() -> None:
    """Ask endpoint should validate payloads and return agent responses."""

    app.dependency_overrides[get_healthcare_agent] = lambda: FakeAgent()
    try:
        response = TestClient(app).post(
            "/api/v1/ask",
            json={"question": "What is the policy?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "received: What is the policy?"
