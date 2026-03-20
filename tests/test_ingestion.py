"""Tests for ingestion engine."""

from pathlib import Path

from mcp_server.ingestion import (
    _chunk_text,
    _is_text_file,
    _load_gitignore,
)


def test_chunk_text_small():
    """Text smaller than chunk_size should return as single chunk."""
    text = "Hello, world!"
    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    """Empty text should return empty list."""
    chunks = _chunk_text("", chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_chunk_text_splits():
    """Long text should be split into multiple chunks."""
    text = "\n\n".join([f"Paragraph {i}. " * 10 for i in range(20)])
    chunks = _chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        # Chunks may slightly exceed size due to overlap
        assert len(chunk) < 500


def test_is_text_file():
    """Should recognize common text file extensions."""
    assert _is_text_file(Path("main.py")) is True
    assert _is_text_file(Path("app.js")) is True
    assert _is_text_file(Path("config.yaml")) is True
    assert _is_text_file(Path("README.md")) is True
    assert _is_text_file(Path("query.sql")) is True
    assert _is_text_file(Path("Dockerfile")) is True


def test_is_not_text_file():
    """Should reject binary file extensions."""
    assert _is_text_file(Path("image.png")) is False
    assert _is_text_file(Path("video.mp4")) is False
    assert _is_text_file(Path("archive.zip")) is False
    assert _is_text_file(Path("data.sqlite3")) is False


def test_load_gitignore(tmp_codebase):
    """Should load .gitignore patterns."""
    spec = _load_gitignore(str(tmp_codebase))
    assert spec is not None
    assert spec.match_file("__pycache__/foo.pyc")
    assert spec.match_file("node_modules/package.json")
    assert spec.match_file(".env")
    assert not spec.match_file("main.py")


def test_load_gitignore_missing(tmp_path):
    """Should return None when no .gitignore exists."""
    spec = _load_gitignore(str(tmp_path))
    assert spec is None
