"""Document loading utilities."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from core.config.settings import Settings, get_settings
from core.exceptions import DocumentLoadError
from schemas import Document, DocumentMetadata


class DocumentLoader:
    """Load healthcare knowledge documents from files and directories."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supported_extensions = {
            extension.lower() for extension in self.settings.SUPPORTED_DOCUMENT_EXTENSIONS
        }

    def load(
        self,
        input_path: Path | None = None,
        *,
        document_type: str | None = None,
    ) -> list[Document]:
        """Load all supported documents from a file or directory."""

        resolved_path = input_path or self.settings.RAW_DATA_DIR
        if not resolved_path.exists():
            raise DocumentLoadError(
                "Input path does not exist.",
                details={"path": str(resolved_path)},
            )

        files = [resolved_path] if resolved_path.is_file() else self._discover_files(resolved_path)
        documents: list[Document] = []
        for file_path in files:
            content = self._read_file(file_path)
            if not content.strip():
                continue
            documents.append(
                Document(
                    content=content,
                    metadata=DocumentMetadata(
                        document_name=file_path.name,
                        document_type=document_type or self._infer_document_type(file_path),
                    ),
                )
            )
        return documents

    def _discover_files(self, directory: Path) -> list[Path]:
        """Discover supported files under a directory."""

        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self.supported_extensions
        )

    def _read_file(self, file_path: Path) -> str:
        """Read supported file content."""

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise DocumentLoadError(
                "Unsupported document type.",
                details={"path": str(file_path), "extension": suffix},
            )
        try:
            if suffix == ".pdf":
                return self._read_pdf(file_path)
            return file_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise DocumentLoadError(
                "Failed to load document.",
                details={"path": str(file_path), "error": str(exc)},
            ) from exc

    @staticmethod
    def _read_pdf(file_path: Path) -> str:
        """Extract text from a PDF file."""

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(page.strip() for page in pages if page.strip())

    @staticmethod
    def _infer_document_type(file_path: Path) -> str:
        """Infer a document category from the file name."""

        name = file_path.stem.lower()
        if "appointment" in name:
            return "appointment_policy"
        if "telehealth" in name or "virtual" in name:
            return "telehealth_policy"
        if "insurance" in name or "claim" in name:
            return "insurance_faq"
        if "hipaa" in name or "privacy" in name:
            return "hipaa_guidelines"
        if "procedure" in name or "workflow" in name or "instruction" in name:
            return "procedure"
        if "article" in name or "education" in name:
            return "healthcare_article"
        return "general_healthcare_document"
