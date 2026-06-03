"""Lightweight request tracing helpers."""

from __future__ import annotations

from contextvars import ContextVar

from common.utils import generate_id

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def start_trace(prefix: str = "trace") -> str:
    """Create and bind a trace ID for the current context."""

    trace_id = generate_id(prefix)
    _trace_id.set(trace_id)
    return trace_id


def get_trace_id() -> str | None:
    """Return the trace ID for the current context."""

    return _trace_id.get()
