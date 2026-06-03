"""Unit tests for chunking strategies."""

from __future__ import annotations

from core.config.settings import Settings
from rag.ingestion.chunkers.contextual import ContextualChunker
from rag.ingestion.chunkers.hybrid import HybridChunker
from rag.ingestion.chunkers.recursive import RecursiveChunker
from rag.ingestion.chunkers.semantic import SemanticChunker
from schemas import Document, DocumentMetadata


def _settings(tmp_path) -> Settings:
    return Settings(
        PROJECT_ROOT=tmp_path,
        CHUNK_SIZE=120,
        CHUNK_OVERLAP=20,
        CONTEXTUAL_MAX_GROUP_SIZE=180,
        SEMANTIC_SIMILARITY_THRESHOLD=0.10,
    )


def _document(document_type: str) -> Document:
    return Document(
        content=(
            "# Refills\nPatients may request medication refills online. "
            "Telehealth consultations may be required. "
            "Insurance coverage depends on plan rules."
        ),
        metadata=DocumentMetadata(
            document_name="policy.md",
            document_type=document_type,
        ),
    )


def test_recursive_chunker_adds_required_metadata(tmp_path) -> None:
    """Recursive chunking should emit searchable chunks with metadata."""

    chunks = RecursiveChunker(_settings(tmp_path)).chunk(_document("policy"))

    assert chunks
    assert chunks[0].metadata.document_name == "policy.md"
    assert chunks[0].metadata.document_type == "policy"
    assert chunks[0].metadata.chunk_strategy == "recursive"
    assert chunks[0].metadata.chunk_id


def test_semantic_chunker_groups_sentences(tmp_path) -> None:
    """Semantic chunking should work without an embedding fallback."""

    chunks = SemanticChunker(_settings(tmp_path)).chunk(_document("healthcare_article"))

    assert chunks
    assert all(chunk.metadata.chunk_strategy == "semantic" for chunk in chunks)


def test_contextual_chunker_preserves_workflow_groups(tmp_path) -> None:
    """Contextual chunking should keep procedure steps together."""

    document = Document(
        content="Procedure\n1. Verify identity.\n2. Review medication.\n3. Submit refill.",
        metadata=DocumentMetadata(document_name="procedure.md", document_type="procedure"),
    )

    chunks = ContextualChunker(_settings(tmp_path)).chunk(document)

    assert len(chunks) == 1
    assert "Verify identity" in chunks[0].content


def test_hybrid_chunker_routes_by_document_type(tmp_path) -> None:
    """Hybrid chunking should delegate based on document type."""

    chunks = HybridChunker(_settings(tmp_path)).chunk(_document("healthcare_article"))

    assert chunks
    assert chunks[0].metadata.chunk_strategy == "semantic"
