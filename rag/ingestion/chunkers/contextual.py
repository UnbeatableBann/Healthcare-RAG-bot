"""Contextual document chunking."""

from __future__ import annotations

import re

from common.utils import normalize_whitespace, sha256_text
from core.config.settings import Settings
from rag.ingestion.chunkers.base import BaseChunker
from schemas import Document, DocumentMetadata, TextChunk


class ContextualChunker(BaseChunker):
    """Chunk procedures and workflows while preserving related steps."""

    strategy_name = "contextual"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunk(self, document: Document) -> list[TextChunk]:
        """Group headings, instructions, and step lists into coherent chunks."""

        lines = [line.strip() for line in document.content.replace("\r\n", "\n").split("\n")]
        groups: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if not line:
                continue
            starts_new_context = self._is_heading(line) and current
            too_large = len(" ".join(current)) + len(line) > self.settings.CONTEXTUAL_MAX_GROUP_SIZE
            if starts_new_context or too_large:
                groups.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            groups.append(current)

        return [
            self._build_chunk(document, normalize_whitespace(" ".join(group)), index)
            for index, group in enumerate(groups)
            if group
        ]

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Return whether a line looks like a context boundary."""

        return bool(
            re.match(r"^#{1,6}\s+", line)
            or re.match(r"^[A-Z][A-Za-z0-9 /:-]{3,80}:?$", line)
        )

    def _build_chunk(self, document: Document, content: str, index: int) -> TextChunk:
        """Create a chunk with required metadata."""

        chunk_id = sha256_text(
            f"{document.metadata.document_name}:{self.strategy_name}:{index}:{content}"
        )
        return TextChunk(
            content=content,
            metadata=DocumentMetadata(
                document_name=document.metadata.document_name,
                document_type=document.metadata.document_type,
                chunk_strategy=self.strategy_name,
                chunk_id=chunk_id,
                extra=document.metadata.extra,
            ),
        )
