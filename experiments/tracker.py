"""JSON experiment tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from common.utils import read_json, utc_now_iso, write_json
from core.config.settings import Settings, get_settings


@dataclass(frozen=True)
class ExperimentRecord:
    """Experiment result stored in JSON."""

    llm: str
    embedding: str
    reranker: str
    chunker: str
    ragas_metrics: dict[str, float]
    latency_seconds: float
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentTracker:
    """Append-only JSON experiment tracker."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def record(self, record: ExperimentRecord) -> list[dict[str, Any]]:
        """Append an experiment record and return the full history."""

        history = read_json(self.settings.EXPERIMENT_RESULTS_PATH, default=[])
        if not isinstance(history, list):
            history = []
        history.append(asdict(record))
        write_json(self.settings.EXPERIMENT_RESULTS_PATH, history)
        return history
