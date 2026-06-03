"""Base Pydantic schemas shared by API and RAG layers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    """Common Pydantic v2 configuration for platform schemas."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ErrorDetail(BaseSchema):
    """Structured API error detail."""

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic details.",
    )


class ErrorResponse(BaseSchema):
    """Standard API error response."""

    error: ErrorDetail


class HealthResponse(BaseSchema):
    """Health check response."""

    service: str
    version: str
    status: Literal["ok"]
    environment: str
    metrics_enabled: bool


class DocumentMetadata(BaseSchema):
    """Metadata stored with source documents and chunks."""

    document_name: str
    document_type: str
    chunk_strategy: str | None = None
    chunk_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseSchema):
    """Loaded source document."""

    content: str = Field(min_length=1)
    metadata: DocumentMetadata


class TextChunk(BaseSchema):
    """Searchable chunk produced from a source document."""

    content: str = Field(min_length=1)
    metadata: DocumentMetadata
    score: float | None = Field(default=None, ge=0.0)
    reranker_score: float | None = Field(default=None)


class RetrievalResult(BaseSchema):
    """Ranked retrieval result returned by dense, sparse, and hybrid retrieval."""

    chunk: TextChunk
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0)
    source: Literal["dense", "sparse", "hybrid", "reranker"]


class Citation(BaseSchema):
    """Citation metadata returned with grounded answers."""

    document_name: str
    document_type: str
    chunk_id: str
    chunk_strategy: str


class AskRequest(BaseSchema):
    """Question-answering request payload."""

    question: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)


class AskResponse(BaseSchema):
    """Grounded assistant response."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    route: str
    answerable: bool


class IngestRequest(BaseSchema):
    """Document ingestion request payload."""

    input_path: Path | None = Field(
        default=None,
        description="Optional file or directory path. Defaults to RAW_DATA_DIR.",
    )
    document_type: str | None = Field(default=None, max_length=128)
    chunking_strategy: str | None = Field(default=None, max_length=64)


class IngestResponse(BaseSchema):
    """Document ingestion response."""

    documents_loaded: int = Field(ge=0)
    chunks_created: int = Field(ge=0)
    collection_name: str
    chunking_strategy: str

