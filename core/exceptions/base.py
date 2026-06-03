"""Domain exceptions for the Healthcare AI Assistant."""

from __future__ import annotations

from typing import Any


class HealthcareAssistantError(Exception):
    """Base class for all domain-specific application errors."""

    error_code = "HEALTHCARE_ASSISTANT_ERROR"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code or self.error_code
        self.status_code = status_code or self.status_code


class ConfigurationError(HealthcareAssistantError):
    """Raised when runtime configuration is invalid."""

    error_code = "CONFIGURATION_ERROR"
    status_code = 500


class ProviderRegistrationError(HealthcareAssistantError):
    """Raised when a provider cannot be registered."""

    error_code = "PROVIDER_REGISTRATION_ERROR"
    status_code = 500


class ProviderNotFoundError(HealthcareAssistantError):
    """Raised when a configured provider is not registered."""

    error_code = "PROVIDER_NOT_FOUND"
    status_code = 500


class ExternalServiceError(HealthcareAssistantError):
    """Raised when an external provider or service fails."""

    error_code = "EXTERNAL_SERVICE_ERROR"
    status_code = 502


class DocumentLoadError(HealthcareAssistantError):
    """Raised when a document cannot be loaded from disk."""

    error_code = "DOCUMENT_LOAD_ERROR"
    status_code = 400


class IngestionError(HealthcareAssistantError):
    """Raised when the ingestion pipeline fails."""

    error_code = "INGESTION_ERROR"
    status_code = 500


class RetrievalError(HealthcareAssistantError):
    """Raised when retrieval cannot be completed."""

    error_code = "RETRIEVAL_ERROR"
    status_code = 500


class GenerationError(HealthcareAssistantError):
    """Raised when answer generation fails."""

    error_code = "GENERATION_ERROR"
    status_code = 502

