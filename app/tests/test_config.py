from __future__ import annotations

from pathlib import Path

from app.config import get_settings


def test_settings_default_pdfs_dir_uses_workspace_pdfs_folder() -> None:
    settings = get_settings()
    expected = Path(__file__).resolve().parents[1] / "pdfs"
    assert settings.pdfs_dir == expected
