"""Document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from common.utils import read_json, write_json
from core.config.settings import Settings, get_settings
from core.exceptions import IngestionError
from embeddings.base import BaseEmbedding
from embeddings.factory import EmbeddingFactory
from rag.ingestion.chunkers.base import BaseChunker
from rag.ingestion.chunkers.factory import ChunkerFactory
from rag.ingestion.document_loader import DocumentLoader
from schemas import IngestResponse, TextChunk
from vectorstores.base import BaseVectorStore
from vectorstores.factory import VectorStoreFactory


class IngestPipeline:
    """Load, chunk, embed, store, and persist healthcare documents."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        loader: DocumentLoader | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or DocumentLoader(self.settings)
        self.embedding = embedding or EmbeddingFactory.create(settings=self.settings)
        self.vector_store = vector_store or VectorStoreFactory.create(settings=self.settings)

    def run(
        self,
        *,
        input_path: Path | None = None,
        document_type: str | None = None,
        chunking_strategy: str | None = None,
    ) -> IngestResponse:
        """Execute document ingestion."""

        try:
            documents = self.loader.load(input_path, document_type=document_type)
            chunker = self._build_chunker(chunking_strategy)
            chunks = [
                chunk
                for document in documents
                for chunk in chunker.chunk(document)
                if chunk.content.strip()
            ]
            if chunks:
                vectors = self.embedding.embed_texts([chunk.content for chunk in chunks])
                self.vector_store.ensure_collection()
                self.vector_store.upsert(chunks, vectors)
                self._persist_chunks(chunks)
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(
                "Document ingestion failed.",
                details={"error": str(exc)},
            ) from exc

        return IngestResponse(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            collection_name=self.settings.QDRANT_COLLECTION_NAME,
            chunking_strategy=chunker.strategy_name,
        )

    def _build_chunker(self, strategy: str | None) -> BaseChunker:
        """Create the requested chunker."""

        return ChunkerFactory.create(
            strategy,
            settings=self.settings,
            embedding=self.embedding,
        )

    def _persist_chunks(self, chunks: list[TextChunk]) -> None:
        """Persist chunks for BM25 sparse retrieval."""

        existing_payload = read_json(self.settings.PROCESSED_CHUNKS_PATH, default=[])
        existing_chunks = [
            TextChunk.model_validate(item)
            for item in existing_payload
            if isinstance(item, dict)
        ]
        by_id = {
            chunk.metadata.chunk_id: chunk
            for chunk in existing_chunks + chunks
            if chunk.metadata.chunk_id
        }
        write_json(
            self.settings.PROCESSED_CHUNKS_PATH,
            [chunk.model_dump(mode="json") for chunk in by_id.values()],
        )
