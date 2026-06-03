"""CLI entrypoint for document ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.ingestion.ingest_pipeline import IngestPipeline


def main() -> None:
    """Run ingestion from the command line."""

    parser = argparse.ArgumentParser(description="Ingest healthcare documents.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--document-type", default=None)
    parser.add_argument("--chunking-strategy", default=None)
    args = parser.parse_args()

    result = IngestPipeline().run(
        input_path=args.input_path,
        document_type=args.document_type,
        chunking_strategy=args.chunking_strategy,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
