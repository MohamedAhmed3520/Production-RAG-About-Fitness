from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATHS = [CURRENT_DIR / ".env", CURRENT_DIR.parent / ".env"]
for env_path in ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path, override=True)
        break
else:
    print("WARNING: No .env file found.")


@dataclass(slots=True)
class Settings:
    """Application configuration."""

    project_root: Path = field(default_factory=lambda: CURRENT_DIR)
    data_dir: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "data")
    cache_dir: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "cache")
    logs_dir: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "logs")
    prompts_dir: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "prompts")
    models_dir: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "models")
    vector_store_path: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "cache" / "faiss.index")
    metadata_path: Path = field(default_factory=lambda: CURRENT_DIR / "medical_agentic_rag" / "cache" / "metadata.json")
    pdfs_dir: Path = field(default_factory=lambda: CURRENT_DIR.parent / "pdfs")
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-large"
    temperature: float = 0.1
    max_tokens: int = 800
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    hybrid_alpha: float = 0.6
    enable_async: bool = True

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings(
            project_root=CURRENT_DIR,
            data_dir=Path(os.getenv("DATA_DIR", CURRENT_DIR / "medical_agentic_rag" / "data")),
            cache_dir=Path(os.getenv("CACHE_DIR", CURRENT_DIR / "medical_agentic_rag" / "cache")),
            logs_dir=Path(os.getenv("LOGS_DIR", CURRENT_DIR / "medical_agentic_rag" / "logs")),
            prompts_dir=Path(os.getenv("PROMPTS_DIR", CURRENT_DIR / "medical_agentic_rag" / "prompts")),
            models_dir=Path(os.getenv("MODELS_DIR", CURRENT_DIR / "medical_agentic_rag" / "models")),
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", CURRENT_DIR / "medical_agentic_rag" / "cache" / "faiss.index")),
            metadata_path=Path(os.getenv("METADATA_PATH", CURRENT_DIR / "medical_agentic_rag" / "cache" / "metadata.json")),
            pdfs_dir=Path(os.getenv("PDFS_DIR", str(CURRENT_DIR.parent / "pdfs"))),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large"),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "700")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("TOP_K", "8")),
            hybrid_alpha=float(os.getenv("HYBRID_ALPHA", "0.6")),
            enable_async=os.getenv("ENABLE_ASYNC", "true").lower() == "true",
        )
        if not _SETTINGS.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing. Check your .env file.")
    return _SETTINGS
