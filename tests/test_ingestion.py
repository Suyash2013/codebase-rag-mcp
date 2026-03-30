"""Tests for ingestion engine."""

from pathlib import Path

from mcp_server.ingestion import (
    TEXT_FILENAMES,
    _collect_text_files,
    _is_text_file,
    _load_gitignore,
    _normalise_extensions,
)
from mcp_server.chunkers.recursive import RecursiveChunker

# Compatibility shim: old _chunk_text tests now use RecursiveChunker
_rc = RecursiveChunker()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Shim for tests that were written against the old ingestion._chunk_text."""
    chunks = _rc.chunk(text, chunk_size, chunk_overlap)
    return [c.text for c in chunks]


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
    assert (
        _is_text_file(
            Path("a.py"),
            include_extensions={".py", ".md"},
            exclude_extensions={".md"},
        )
        is True
    )
    # .md in include AND in exclude → rejected
    assert (
        _is_text_file(
            Path("README.md"),
            include_extensions={".py", ".md"},
            exclude_extensions={".md"},
        )
        is False
    )


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


# ---------------------------------------------------------------------------
# _chunk_text edge-case tests
# ---------------------------------------------------------------------------


def test_chunk_text_exactly_chunk_size():
    """Text exactly at chunk_size should be returned as a single chunk."""
    text = "x" * 100
    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == [text]


def test_chunk_text_whitespace_only():
    """Whitespace-only text should return an empty list."""
    chunks = _chunk_text("   \n\n  \t  ", chunk_size=50, chunk_overlap=10)
    assert chunks == []


def test_chunk_text_no_overlap():
    """With chunk_overlap=0 chunks should not contain content from adjacent chunks."""
    # Use double-newline separators so we know how splits happen
    para_a = "A" * 60
    para_b = "B" * 60
    text = para_a + "\n\n" + para_b
    chunks = _chunk_text(text, chunk_size=80, chunk_overlap=0)
    assert len(chunks) >= 2
    # No chunk should start with content from a different paragraph
    assert not any(chunk.startswith("A") and "B" in chunk for chunk in chunks)
    assert not any(chunk.startswith("B") and "A" in chunk for chunk in chunks)


def test_chunk_text_double_newline_separator():
    """Text split on \\n\\n should produce chunks bounded by paragraphs."""
    para_a = "Word " * 15   # ~75 chars
    para_b = "Text " * 15   # ~75 chars
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=0)
    # Should produce exactly 2 chunks, one per paragraph
    assert len(chunks) == 2
    assert para_a.strip() in chunks[0]
    assert para_b.strip() in chunks[1]


def test_chunk_text_newline_separator():
    """Text split on single \\n (no \\n\\n present) should use newline boundary."""
    line_a = "Line_A " * 10   # ~70 chars
    line_b = "Line_B " * 10   # ~70 chars
    text = line_a.strip() + "\n" + line_b.strip()
    chunks = _chunk_text(text, chunk_size=90, chunk_overlap=0)
    assert len(chunks) == 2
    assert "Line_A" in chunks[0]
    assert "Line_B" in chunks[1]


def test_chunk_text_overlap_correctness():
    """Each chunk after the first should start with the tail of the previous chunk."""
    para_a = "Alpha " * 15   # ~90 chars
    para_b = "Beta " * 15    # ~75 chars
    text = para_a.strip() + "\n\n" + para_b.strip()
    overlap = 20
    chunks = _chunk_text(text, chunk_size=110, chunk_overlap=overlap)
    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        expected_prefix = chunks[i - 1][-overlap:]
        assert chunks[i].startswith(expected_prefix), (
            f"Chunk {i} does not start with tail of chunk {i-1}. "
            f"Expected prefix: {expected_prefix!r}, got start: {chunks[i][:overlap]!r}"
        )


def test_chunk_text_space_separator():
    """Text with no newlines falls back to space separator."""
    # Single long line, no newlines
    words = ["word"] * 40   # "word word word ..." ~200 chars
    text = " ".join(words)
    chunks = _chunk_text(text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) > 1
    # Every chunk should contain recognisable word content
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_text_hard_split_base_case():
    """Text with no separators at all must be hard-split at chunk_size.

    The hard-split stride is (chunk_size - chunk_overlap), and the post-hoc
    overlap pass prepends chunk_overlap chars from the previous chunk, so
    final chunk length is at most chunk_size + chunk_overlap.
    """
    text = "X" * 250
    chunk_size = 100
    chunk_overlap = 10
    chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1
    # Each chunk is at most chunk_size + chunk_overlap due to double-overlap
    for chunk in chunks:
        assert len(chunk) <= chunk_size + chunk_overlap


def test_chunk_text_multiple_paragraphs_overlap():
    """Multiple paragraphs with overlap: verify chunk count and overlap prefix."""
    # Build 4 paragraphs each ~80 chars wide, chunk_size=100 so each para is its own chunk
    paragraphs = ["Para%d " % i * 13 for i in range(4)]   # each ~91 chars
    text = "\n\n".join(p.strip() for p in paragraphs)
    overlap = 15
    chunks = _chunk_text(text, chunk_size=100, chunk_overlap=overlap)
    assert len(chunks) == 4
    for i in range(1, len(chunks)):
        expected_prefix = chunks[i - 1][-overlap:]
        assert chunks[i].startswith(expected_prefix)
