from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
