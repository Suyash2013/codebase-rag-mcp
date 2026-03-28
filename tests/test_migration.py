"""Tests for data directory migration from .codebase-rag to .rag-mcp."""

from mcp_server.migration import migrate_data_directory


def test_migrate_data_dir_renames_old_to_new(tmp_path):
    """If .codebase-rag/ exists and .rag-mcp/ doesn't, rename it."""
    old_dir = tmp_path / ".codebase-rag"
    old_dir.mkdir()
    (old_dir / "test.txt").write_text("data")

    migrate_data_directory(str(tmp_path))

    assert not old_dir.exists()
    assert (tmp_path / ".rag-mcp" / "test.txt").exists()


def test_migrate_data_dir_noop_when_new_exists(tmp_path):
    """If .rag-mcp/ already exists, do nothing (don't overwrite)."""
    (tmp_path / ".rag-mcp").mkdir()
    (tmp_path / ".codebase-rag").mkdir()

    migrate_data_directory(str(tmp_path))

    assert (tmp_path / ".codebase-rag").exists()  # old untouched
    assert (tmp_path / ".rag-mcp").exists()


def test_migrate_data_dir_noop_when_no_old(tmp_path):
    """If .codebase-rag/ doesn't exist, do nothing."""
    migrate_data_directory(str(tmp_path))

    assert not (tmp_path / ".rag-mcp").exists()
