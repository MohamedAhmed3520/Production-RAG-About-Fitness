from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from rank_bm25 import BM25Okapi

from rag.loaders.pdf_loader import DocumentChunk


class HybridRetriever:
    """Combine dense vector search with BM25 over indexed chunks."""

    def __init__(self, vector_store: object, embedding_service: object) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self._documents: List[DocumentChunk] = []
        self._bm25 = None

    def index_documents(self, documents: Sequence[DocumentChunk]) -> None:
        self._documents = list(documents)
        self.vector_store.clear_collection()
        tokenized = [self._tokenize(doc.text) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        points = []
        for doc in self._documents:
            embedding = self.embedding_service.embed_text(doc.text)
            points.append({"embedding": embedding, "payload": {**doc.metadata, "text": doc.text}})
        self.vector_store.upsert_points(points)

    def retrieve(self, query: str, top_k: int = 30) -> List[Tuple[Dict, float]]:
        if not self._documents:
            return []
        dense_query = self.embedding_service.embed_text(query)
        dense_results = self.vector_store.search(dense_query, limit=top_k)
        bm25_scores = self._bm25.get_scores(self._tokenize(query))
        combined: List[Tuple[Dict, float]] = []
        for result in dense_results:
            payload = dict(result)
            doc_text = payload.get("text", "")
            idx = self._find_index(doc_text)
            if idx is None:
                continue
            bm25_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
            combined.append((payload, float(payload.get("score", 0.0)) + bm25_score * 0.05))
        combined.sort(key=lambda item: item[1], reverse=True)
        return combined[:top_k]

    def _find_index(self, text: str) -> int | None:
        for idx, doc in enumerate(self._documents):
            if doc.text == text:
                return idx
        return None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())
