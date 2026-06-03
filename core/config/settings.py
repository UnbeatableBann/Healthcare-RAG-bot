"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]
ProviderName = Literal["ollama", "vllm", "llamacpp"]
EmbeddingProviderName = Literal["bge", "e5", "nomic"]
RerankerProviderName = Literal["bge", "jina"]
VectorStoreProviderName = Literal["qdrant"]
ChunkingStrategyName = Literal["recursive", "semantic", "contextual", "hybrid"]


class Settings(BaseSettings):
    """Typed runtime configuration for the full platform."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2],
        description="Absolute path to the project root.",
    )

    APP_NAME: str = Field(
        default="Healthcare AI Assistant",
        description="Human-readable application name.",
    )
    APP_VERSION: str = Field(default="0.1.0", description="Application version.")
    ENVIRONMENT: Environment = Field(
        default="development",
        description="Runtime environment.",
    )
    API_PREFIX: str = Field(default="/api/v1", description="Versioned API prefix.")
    HOST: str = Field(default="0.0.0.0", description="API bind host.")
    PORT: PositiveInt = Field(default=8000, description="API bind port.")
    CORS_ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins.",
    )

    LOG_LEVEL: str = Field(default="INFO", description="Application log level.")
    LOG_JSON: bool = Field(default=False, description="Emit structured JSON logs.")

    METRICS_ENABLED: bool = Field(default=True, description="Enable Prometheus metrics.")

    DATA_DIR: Path = Field(default=Path("data"), description="Data root directory.")
    RAW_DATA_DIR: Path = Field(
        default=Path("data/raw"),
        description="Raw document directory.",
    )
    PROCESSED_DATA_DIR: Path = Field(
        default=Path("data/processed"),
        description="Processed document directory.",
    )
    EVALUATION_DATA_DIR: Path = Field(
        default=Path("data/evaluation"),
        description="Evaluation dataset directory.",
    )
    PROCESSED_CHUNKS_PATH: Path = Field(
        default=Path("data/processed/chunks.json"),
        description="Persisted chunk corpus used by sparse retrieval.",
    )
    SUPPORTED_DOCUMENT_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf"],
        description="Document file extensions accepted by the loader.",
    )
    EXPERIMENT_RESULTS_PATH: Path = Field(
        default=Path("experiments/results.json"),
        description="JSON experiment tracking file.",
    )

    LLM_PROVIDER: ProviderName = Field(
        default="ollama",
        description="Configured LLM provider.",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama HTTP API base URL.",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.1",
        description="Default Ollama model.",
    )
    VLLM_BASE_URL: str = Field(
        default="http://localhost:8001",
        description="vLLM OpenAI-compatible API base URL.",
    )
    VLLM_MODEL: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Default vLLM model identifier.",
    )
    LLAMACPP_BASE_URL: str = Field(
        default="http://localhost:8080",
        description="llama.cpp server base URL.",
    )
    LLAMACPP_MODEL: str = Field(
        default="local-llamacpp-model",
        description="Default llama.cpp model label.",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    LLM_MAX_TOKENS: PositiveInt = Field(
        default=1024,
        description="Maximum generation tokens.",
    )
    LLM_REQUEST_TIMEOUT_SECONDS: PositiveFloat = Field(
        default=60.0,
        description="LLM request timeout.",
    )

    EMBEDDING_PROVIDER: EmbeddingProviderName = Field(
        default="bge",
        description="Configured embedding provider.",
    )
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name.",
    )
    EMBEDDING_BATCH_SIZE: PositiveInt = Field(
        default=32,
        description="Embedding batch size.",
    )
    EMBEDDING_DEVICE: str | None = Field(
        default=None,
        description="Optional sentence-transformers device override.",
    )

    RERANKER_PROVIDER: RerankerProviderName = Field(
        default="bge",
        description="Configured reranker provider.",
    )
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-base",
        description="Reranker model name.",
    )
    RERANKER_TOP_N: PositiveInt = Field(
        default=10,
        description="Number of chunks retained after reranking.",
    )
    FINAL_CONTEXT_TOP_K: PositiveInt = Field(
        default=5,
        description="Number of chunks passed to generation.",
    )
    RERANKER_REQUEST_TIMEOUT_SECONDS: PositiveFloat = Field(
        default=30.0,
        description="Reranker request timeout.",
    )
    JINA_API_KEY: str | None = Field(
        default=None,
        description="Optional Jina API key for hosted reranking.",
    )
    JINA_RERANKER_URL: str = Field(
        default="https://api.jina.ai/v1/rerank",
        description="Jina hosted reranker endpoint.",
    )

    VECTORSTORE_PROVIDER: VectorStoreProviderName = Field(
        default="qdrant",
        description="Configured vector store provider.",
    )
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP API base URL.",
    )
    QDRANT_API_KEY: str | None = Field(
        default=None,
        description="Optional Qdrant API key.",
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="healthcare_documents",
        description="Qdrant collection name.",
    )
    QDRANT_TIMEOUT_SECONDS: PositiveFloat = Field(
        default=30.0,
        description="Qdrant request timeout.",
    )
    VECTOR_SIZE: PositiveInt = Field(
        default=384,
        description="Dense embedding vector size.",
    )

    DEFAULT_CHUNKING_STRATEGY: ChunkingStrategyName = Field(
        default="hybrid",
        description="Default document chunking strategy.",
    )
    CHUNK_SIZE: PositiveInt = Field(
        default=900,
        description="Maximum chunk size in characters.",
    )
    CHUNK_OVERLAP: int = Field(
        default=120,
        ge=0,
        description="Chunk overlap in characters.",
    )
    SEMANTIC_SIMILARITY_THRESHOLD: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Adjacent segment similarity threshold for semantic chunking.",
    )
    CONTEXTUAL_MAX_GROUP_SIZE: PositiveInt = Field(
        default=1400,
        description="Maximum character size for contextual procedure groups.",
    )

    MULTI_QUERY_COUNT: PositiveInt = Field(
        default=3,
        description="Number of generated query variants.",
    )
    QUERY_REWRITE_ENABLED: bool = Field(
        default=True,
        description="Enable LLM-assisted query rewriting.",
    )
    DENSE_TOP_K: PositiveInt = Field(
        default=30,
        description="Dense retrieval candidate count.",
    )
    SPARSE_TOP_K: PositiveInt = Field(
        default=30,
        description="Sparse retrieval candidate count.",
    )
    HYBRID_TOP_K: PositiveInt = Field(
        default=30,
        description="Hybrid retrieval candidate count before reranking.",
    )
    RRF_K: PositiveInt = Field(
        default=60,
        description="Reciprocal Rank Fusion smoothing constant.",
    )

    CRAG_LOW_CONFIDENCE_THRESHOLD: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="CRAG threshold below which generation is rejected.",
    )
    CRAG_HIGH_CONFIDENCE_THRESHOLD: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="CRAG threshold above which generation proceeds.",
    )
    CRAG_RETRIEVAL_SCORE_WEIGHT: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="CRAG heuristic weight for retrieval score.",
    )
    CRAG_RERANKER_SCORE_WEIGHT: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="CRAG heuristic weight for reranker score.",
    )
    CRAG_CONTEXT_COVERAGE_WEIGHT: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="CRAG heuristic weight for context coverage.",
    )
    LLM_JUDGE_ENABLED: bool = Field(
        default=True,
        description="Enable LLM-as-a-judge for medium-confidence retrieval.",
    )
    REFUSAL_MESSAGE: str = Field(
        default="I could not find this information in the provided documents.",
        description="Grounded refusal returned when retrieved context is insufficient.",
    )

    APPOINTMENT_CLINIC_NAME: str = Field(
        default="Healthcare Knowledge Clinic",
        description="Clinic name used by the appointment tool.",
    )
    APPOINTMENT_HOURS: str = Field(
        default="Monday to Friday, 09:00-17:00",
        description="Appointment availability window exposed by the appointment tool.",
    )
    APPOINTMENT_CONTACT: str = Field(
        default="front-desk@example.org",
        description="Appointment contact channel exposed by the appointment tool.",
    )

    @field_validator("API_PREFIX")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Ensure the API prefix starts with one slash and has no trailing slash."""

        normalized = "/" + value.strip("/")
        if normalized == "/":
            msg = "API_PREFIX must contain a non-root path."
            raise ValueError(msg)
        return normalized

    @field_validator("CORS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Allow CORS origins to be configured as a comma-separated string."""

        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def resolve_runtime_paths(self) -> "Settings":
        """Resolve relative runtime paths from the project root."""

        for field_name in (
            "DATA_DIR",
            "RAW_DATA_DIR",
            "PROCESSED_DATA_DIR",
            "EVALUATION_DATA_DIR",
            "PROCESSED_CHUNKS_PATH",
            "EXPERIMENT_RESULTS_PATH",
        ):
            value = getattr(self, field_name)
            if not value.is_absolute():
                setattr(self, field_name, self.PROJECT_ROOT / value)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()

