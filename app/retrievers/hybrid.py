from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document


class HybridRetriever:
    """Create a lightweight retriever wrapper that works across LangChain versions."""

    def __init__(self, dense_retriever: Any, bm25_retriever: Any | None = None) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.retriever = dense_retriever

    def get_retriever(self) -> Any:
        return self.retriever
