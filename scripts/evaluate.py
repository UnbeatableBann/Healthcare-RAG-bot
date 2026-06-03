"""CLI entrypoint for RAGAS evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.utils import read_json
from evaluation.ragas_runner import RAGASSample, RAGASRunner


def main() -> None:
    """Run RAGAS evaluation from a JSON file."""

    parser = argparse.ArgumentParser(description="Evaluate RAG outputs with RAGAS.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()

    payload = read_json(args.dataset, default=[])
    samples = [RAGASSample(**item) for item in payload]
    metrics = RAGASRunner().evaluate(samples)
    print(metrics)


if __name__ == "__main__":
    main()
