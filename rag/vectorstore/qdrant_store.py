from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


class QdrantVectorStore:
    """Persist and query vector embeddings in Qdrant."""

    def __init__(self, url: str, collection_name: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = self.client.get_collections().collections
        names = [c.name for c in existing]
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            )

    def upsert_points(self, points: Sequence[Dict[str, Any]]) -> None:
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=point["embedding"],
                    payload=point["payload"],
                )
                for point in points
            ],
        )

    def clear_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def search(self, query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [dict(r.payload, score=r.score) for r in results]
