"""Chunk deduplication."""

from __future__ import annotations

from schemas import TextChunk


class ChunkDeduplicator:
    """Remove duplicate chunks using `chunk_id`."""

    def deduplicate(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Return unique chunks in first-seen order."""

        seen: set[str] = set()
        unique: list[TextChunk] = []
        for chunk in chunks:
            chunk_id = chunk.metadata.chunk_id or chunk.content
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unique.append(chunk)
        return unique
