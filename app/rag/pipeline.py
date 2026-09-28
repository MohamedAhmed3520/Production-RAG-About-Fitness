from __future__ import annotations

from pathlib import Path
from typing import List

from app.config import get_settings
from app.utils import ensure_directory
from rag.ingestion import PDFIndexer
from rag.loaders.pdf_loader import DocumentChunk, PDFLoader
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.vectorstore.qdrant_store import QdrantVectorStore


class AppRAGPipeline:
    """Thin adapter for indexing and retrieving PDFs from the app package."""

    def __init__(self, pdfs_dir: Path | None = None) -> None:
        settings = get_settings()
        self.pdfs_dir = pdfs_dir or settings.pdfs_dir
        ensure_directory(self.pdfs_dir)
        self.vector_store = QdrantVectorStore(collection_name="fitness_nutrition")
        self.embedding_service = None
        self.retriever = HybridRetriever(self.vector_store, self.embedding_service)
        self.indexer = PDFIndexer(self.pdfs_dir, self.vector_store, self.retriever)

    def index_documents(self) -> List[DocumentChunk]:
        return self.indexer.index_all()

    def retrieve(self, query: str, top_k: int = 8) -> List[tuple[dict, float]]:
        return self.retriever.retrieve(query, top_k=top_k)
