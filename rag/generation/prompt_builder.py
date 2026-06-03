"""Grounded answer prompt construction."""

from __future__ import annotations

from schemas import TextChunk


class PromptBuilder:
    """Build strict RAG prompts that require context grounding."""

    system_prompt = (
        "You are a healthcare knowledge assistant. Answer strictly from the "
        "provided context. If the context does not support the answer, say that "
        "the information could not be found in the provided documents. Do not "
        "use outside medical knowledge. Do not provide diagnosis or emergency "
        "medical advice."
    )

    def build(self, question: str, chunks: list[TextChunk]) -> tuple[str, str]:
        """Return system and user prompts."""

        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                "\n".join(
                    [
                        f"[{index}] Document: {chunk.metadata.document_name}",
                        f"Type: {chunk.metadata.document_type}",
                        f"Chunk ID: {chunk.metadata.chunk_id}",
                        chunk.content,
                    ]
                )
            )
        prompt = (
            "Use the context below to answer the question. Include citation "
            "markers like [1] when referencing context.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{'\n\n'.join(context_blocks)}"
        )
        return self.system_prompt, prompt
