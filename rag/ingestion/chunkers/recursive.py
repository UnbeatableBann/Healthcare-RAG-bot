"""Recursive document chunking."""

from __future__ import annotations

import re

from common.utils import normalize_whitespace, sha256_text
from core.config.settings import Settings
from rag.ingestion.chunkers.base import BaseChunker
from schemas import Document, DocumentMetadata, TextChunk


class RecursiveChunker(BaseChunker):
    """Structure-preserving recursive chunker for policies, FAQs, and guidelines."""

    strategy_name = "recursive"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chunk(self, document: Document) -> list[TextChunk]:
        """Split a document by headings, paragraphs, and sentences."""

        sections = self._split_sections(document.content)
        candidate_units: list[str] = []
        for section in sections:
            candidate_units.extend(self._split_large_unit(section))
        return self._pack_units(candidate_units, document)

    def _split_sections(self, text: str) -> list[str]:
        """Split by markdown headings and paragraph boundaries."""

        normalized = text.replace("\r\n", "\n")
        heading_parts = re.split(r"(?m)(?=^#{1,6}\s+)", normalized)
        sections: list[str] = []
        for part in heading_parts:
            sections.extend(segment.strip() for segment in re.split(r"\n{2,}", part))
        return [section for section in sections if section]

    def _split_large_unit(self, unit: str) -> list[str]:
        """Split oversized units into sentence-level pieces."""

        if len(unit) <= self.settings.CHUNK_SIZE:
            return [normalize_whitespace(unit)]
        sentences = re.split(r"(?<=[.!?])\s+", normalize_whitespace(unit))
        return [sentence for sentence in sentences if sentence]

    def _pack_units(self, units: list[str], document: Document) -> list[TextChunk]:
        """Pack text units into chunks with configured overlap."""

        chunks: list[str] = []
        current = ""
        for unit in units:
            if not current:
                current = unit
                continue
            if len(current) + 1 + len(unit) <= self.settings.CHUNK_SIZE:
                current = f"{current} {unit}"
            else:
                chunks.append(current)
                overlap = current[-self.settings.CHUNK_OVERLAP :] if self.settings.CHUNK_OVERLAP else ""
                current = normalize_whitespace(f"{overlap} {unit}")
        if current:
            chunks.append(current)
        return [
            self._build_chunk(document, content, index)
            for index, content in enumerate(chunks)
            if content
        ]

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
