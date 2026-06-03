"""Prometheus metric definitions for the platform."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total API requests.",
    ["method", "path", "status_code"],
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries.",
    ["outcome"],
)

TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total agent tool calls.",
    ["tool_name", "outcome"],
)

HALLUCINATION_PREVENTED_TOTAL = Counter(
    "hallucination_prevented_total",
    "Total responses refused because context was not sufficient.",
)

API_LATENCY_SECONDS = Histogram(
    "api_latency_seconds",
    "API request latency in seconds.",
    ["method", "path"],
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Retrieval latency in seconds.",
    ["retriever"],
)

RERANKING_LATENCY_SECONDS = Histogram(
    "reranking_latency_seconds",
    "Reranking latency in seconds.",
    ["reranker"],
)

GENERATION_LATENCY_SECONDS = Histogram(
    "generation_latency_seconds",
    "LLM answer generation latency in seconds.",
    ["provider", "model"],
)

AGENT_LATENCY_SECONDS = Histogram(
    "agent_latency_seconds",
    "Agent routing and execution latency in seconds.",
    ["route"],
)


@contextmanager
def observe_latency(metric: Histogram, **labels: str) -> Iterator[None]:
    """Observe elapsed time for a Prometheus histogram."""

    started_at = perf_counter()
    try:
        yield
    finally:
        metric.labels(**labels).observe(perf_counter() - started_at)


def render_prometheus_metrics() -> bytes:
    """Render all registered Prometheus metrics."""

    return generate_latest()

