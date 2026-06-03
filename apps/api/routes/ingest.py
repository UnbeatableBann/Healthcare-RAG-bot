"""Document ingestion route."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from fastapi import APIRouter, Depends

from core.config.settings import Settings, get_settings
from rag.ingestion.ingest_pipeline import IngestPipeline
from schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@lru_cache(maxsize=1)
def get_ingest_pipeline() -> IngestPipeline:
    """Build and cache the ingestion pipeline."""

    settings = get_settings()
    return IngestPipeline(settings=settings)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    pipeline: IngestPipeline = Depends(get_ingest_pipeline),
    _settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Ingest documents into Qdrant and the BM25 corpus."""

    return await asyncio.to_thread(
        pipeline.run,
        input_path=request.input_path,
        document_type=request.document_type,
        chunking_strategy=request.chunking_strategy,
    )
