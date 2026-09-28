from __future__ import annotations

from app.utils import ensure_directory, normalize_text


def test_utility_helpers() -> None:
    path = ensure_directory("/tmp/fitness_assistant_test")
    assert path.exists()
    assert normalize_text(None) == ""
    assert normalize_text("  hello  ") == "hello"
