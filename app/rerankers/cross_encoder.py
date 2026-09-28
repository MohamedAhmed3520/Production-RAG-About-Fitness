from __future__ import annotations

from typing import Any


class RerankerService:
    """Wrap reranking behavior with a compatibility fallback for newer LangChain versions."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self.reranker = None

    def as_compressor(self) -> Any:
        return self.reranker
