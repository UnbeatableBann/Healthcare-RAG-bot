"""Observability utilities."""

from core.observability.metrics import (
    AGENT_LATENCY_SECONDS,
    API_LATENCY_SECONDS,
    API_REQUESTS_TOTAL,
    GENERATION_LATENCY_SECONDS,
    HALLUCINATION_PREVENTED_TOTAL,
    RAG_QUERIES_TOTAL,
    RERANKING_LATENCY_SECONDS,
    RETRIEVAL_LATENCY_SECONDS,
    TOOL_CALLS_TOTAL,
    observe_latency,
    render_prometheus_metrics,
)

__all__ = [
    "AGENT_LATENCY_SECONDS",
    "API_LATENCY_SECONDS",
    "API_REQUESTS_TOTAL",
    "GENERATION_LATENCY_SECONDS",
    "HALLUCINATION_PREVENTED_TOTAL",
    "RAG_QUERIES_TOTAL",
    "RERANKING_LATENCY_SECONDS",
    "RETRIEVAL_LATENCY_SECONDS",
    "TOOL_CALLS_TOTAL",
    "observe_latency",
    "render_prometheus_metrics",
]

