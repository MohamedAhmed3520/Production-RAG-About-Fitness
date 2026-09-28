from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    reranker_model: str
    pdfs_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.5:free"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "fitness_nutrition"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            pdfs_dir=Path(os.getenv("PDFS_DIR", "./pdfs")).resolve(),
        )


settings = Settings.from_env()


def get_settings() -> Settings:
    return settings
