"""Tests for ingestion engine."""

from pathlib import Path

from mcp_server.ingestion import (
    TEXT_FILENAMES,
    _chunk_text,
    _collect_text_files,
    _is_text_file,
    _load_gitignore,
    _normalise_extensions,
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


# ---------------------------------------------------------------------------
# _normalise_extensions
# ---------------------------------------------------------------------------


def test_normalise_extensions_none():
    """None input returns None."""
    assert _normalise_extensions(None) is None


def test_normalise_extensions_empty():
    """Empty list returns None."""
    assert _normalise_extensions([]) is None


def test_normalise_extensions_adds_dot():
    """Extensions without leading dots are normalised."""
    result = _normalise_extensions(["py", "js", ".ts"])
    assert result == {".py", ".js", ".ts"}


def test_normalise_extensions_lowercases():
    """Extensions are lowercased."""
    result = _normalise_extensions([".PY", "JS"])
    assert result == {".py", ".js"}


# ---------------------------------------------------------------------------
# _is_text_file with filters
# ---------------------------------------------------------------------------


def test_is_text_file_include_extensions_allows():
    """`include_extensions` allowlist: matching file is accepted."""
    assert _is_text_file(Path("main.py"), include_extensions={".py"}) is True


def test_is_text_file_include_extensions_blocks():
    """`include_extensions` allowlist: non-matching file is rejected."""
    assert _is_text_file(Path("config.yaml"), include_extensions={".py"}) is False


def test_is_text_file_include_extensions_special_filename():
    """Special filenames (e.g. Dockerfile) pass through even with include_extensions."""
    assert _is_text_file(Path("Dockerfile"), include_extensions={".py"}) is True


def test_is_text_file_exclude_extensions_blocks():
    """`exclude_extensions` blocklist: matching file is rejected."""
    assert _is_text_file(Path("README.md"), exclude_extensions={".md"}) is False


def test_is_text_file_exclude_does_not_block_other():
    """`exclude_extensions` blocklist: non-matching file still passes."""
    assert _is_text_file(Path("main.py"), exclude_extensions={".md"}) is True


def test_is_text_file_include_and_exclude():
    """include_extensions takes priority; exclude_extensions further restricts."""
    # .py in include but not in exclude → accepted
    assert _is_text_file(Path("a.py"), include_extensions={".py", ".md"}, exclude_extensions={".md"}) is True
    # .md in include AND in exclude → rejected
    assert _is_text_file(Path("README.md"), include_extensions={".py", ".md"}, exclude_extensions={".md"}) is False


# ---------------------------------------------------------------------------
# _collect_text_files with filters
# ---------------------------------------------------------------------------


def test_collect_text_files_include_extensions(tmp_codebase):
    """Only files with the specified extensions are collected."""
    files = _collect_text_files(str(tmp_codebase), include_extensions={".py"})
    rel_paths = [rel for _, rel in files]
    # Special filenames (e.g. .gitignore) bypass include_extensions by design
    non_special = [p for p in rel_paths if not any(p.endswith(n) for n in TEXT_FILENAMES)]
    assert all(p.endswith(".py") for p in non_special), rel_paths
    assert any("main.py" in p for p in rel_paths)
    # JS and YAML should be excluded
    assert not any(p.endswith(".js") for p in rel_paths)
    assert not any(p.endswith(".yaml") for p in rel_paths)


def test_collect_text_files_exclude_extensions(tmp_codebase):
    """Files with excluded extensions are not collected."""
    files = _collect_text_files(str(tmp_codebase), exclude_extensions={".md"})
    rel_paths = [rel for _, rel in files]
    assert not any(p.endswith(".md") for p in rel_paths)
    # Other text files should still be present
    assert any(p.endswith(".py") for p in rel_paths)


def test_collect_text_files_no_filters(tmp_codebase):
    """Without filters all default text files are collected."""
    files = _collect_text_files(str(tmp_codebase))
    rel_paths = [rel for _, rel in files]
    assert any(p.endswith(".py") for p in rel_paths)
    assert any(p.endswith(".js") for p in rel_paths)
    assert any(p.endswith(".yaml") for p in rel_paths)
    assert any(p.endswith(".md") for p in rel_paths)
    # Binary PNG should never be collected
    assert not any(p.endswith(".png") for p in rel_paths)
