from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate embeddings for text chunks with caching."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._cache: Dict[str, List[float]] = {}

    def embed_text(self, text: str) -> List[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        embedding = self.model.encode(text, normalize_embeddings=True).tolist()
        self._cache[key] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]
