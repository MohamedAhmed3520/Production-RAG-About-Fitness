from __future__ import annotations

import re
from typing import Any, List

from langchain_core.documents import Document

from app.embeddings.embedding_service import EmbeddingService

try:
    from langchain_community.vectorstores import Qdrant as _Qdrant
except ImportError:  # pragma: no cover - environment specific
    _Qdrant = None


class _InMemoryRetriever:
    """Simple keyword-based retriever used when Qdrant is unavailable."""

    def __init__(self, documents: List[Document], limit: int = 5) -> None:
        self.documents = documents
        self.limit = limit

    def invoke(self, query: str) -> List[Document]:
        if not self.documents:
            return []
        query_terms = {term for term in re.split(r"\W+", query.lower()) if term}
        scored: List[tuple[float, Document]] = []
        for document in self.documents:
            content = (document.page_content or "").lower()
            overlap = sum(1 for term in query_terms if term in content)
            if overlap:
                scored.append((float(overlap), document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[: self.limit]] or self.documents[: self.limit]


class QdrantVectorStoreService:
    """Manage document indexing and retrieval, with an in-memory fallback."""

    def __init__(self, url: str, collection_name: str, embedding_service: EmbeddingService) -> None:
        self.url = url
        self.collection_name = collection_name
        self.embedding_service = embedding_service
        self._documents: List[Document] = []
        self.vector_store = None
        if _Qdrant is not None:
            self.vector_store = _Qdrant.from_documents(
                documents=[],
                embedding=self.embedding_service.embeddings,
                url=url,
                collection_name=collection_name,
                prefer_grpc=False,
            )

    def add_documents(self, docs: List[Document]) -> None:
        self._documents.extend(docs)
        if self.vector_store is None:
            return
        self.vector_store.add_documents(docs)

    def as_retriever(self, search_kwargs: dict | None = None) -> Any:
        if self.vector_store is not None:
            try:
                return self.vector_store.as_retriever(search_kwargs=search_kwargs or {})
            except Exception:
                pass
        limit = (search_kwargs or {}).get("k", 5)
        return _InMemoryRetriever(self._documents, limit=int(limit))
