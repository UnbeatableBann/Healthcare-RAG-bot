"""Semantic document chunking."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from common.utils import normalize_whitespace, sha256_text
from core.config.settings import Settings
from embeddings.base import BaseEmbedding
from rag.ingestion.chunkers.base import BaseChunker
from schemas import Document, DocumentMetadata, TextChunk


class SemanticChunker(BaseChunker):
    """Chunk long-form healthcare content by adjacent semantic similarity."""

    strategy_name = "semantic"

    def __init__(self, settings: Settings, embedding: BaseEmbedding | None = None) -> None:
        self.settings = settings
        self.embedding = embedding

    def chunk(self, document: Document) -> list[TextChunk]:
        """Split a document at topic-boundary-like similarity drops."""

        sentences = self._sentences(document.content)
        if not sentences:
            return []
        similarities = self._adjacent_similarities(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for index, sentence in enumerate(sentences[1:], start=1):
            current_size = len(" ".join(groups[-1])) + len(sentence)
            similarity = similarities[index - 1]
            if (
                similarity < self.settings.SEMANTIC_SIMILARITY_THRESHOLD
                or current_size > self.settings.CHUNK_SIZE
            ):
                groups.append([sentence])
            else:
                groups[-1].append(sentence)
        return [
            self._build_chunk(document, normalize_whitespace(" ".join(group)), index)
            for index, group in enumerate(groups)
            if group
        ]

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split text into sentence-like units."""

        normalized = normalize_whitespace(text)
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]

    def _adjacent_similarities(self, sentences: list[str]) -> list[float]:
        """Compute adjacent sentence similarities."""

        if len(sentences) <= 1:
            return []
        if self.embedding is not None:
            vectors = np.asarray(self.embedding.embed_texts(sentences), dtype=float)
            return [
                self._cosine(vectors[index], vectors[index + 1])
                for index in range(len(sentences) - 1)
            ]
        return [
            self._lexical_cosine(sentences[index], sentences[index + 1])
            for index in range(len(sentences) - 1)
        ]

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        """Return cosine similarity for dense vectors."""

        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    @staticmethod
    def _lexical_cosine(left: str, right: str) -> float:
        """Fallback similarity based on token frequency cosine."""

        left_counts = Counter(re.findall(r"[a-z0-9]+", left.lower()))
        right_counts = Counter(re.findall(r"[a-z0-9]+", right.lower()))
        if not left_counts or not right_counts:
            return 0.0
        vocabulary = set(left_counts) | set(right_counts)
        numerator = sum(left_counts[token] * right_counts[token] for token in vocabulary)
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

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
