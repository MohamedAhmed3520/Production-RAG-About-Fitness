from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingService:
    """Wrap the embedding model used by the vector store."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
