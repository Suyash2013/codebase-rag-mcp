"""Tests for data directory migration to .omni-rag."""

from mcp_server.migration import migrate_data_directory


def test_migrate_data_dir_renames_codebase_rag(tmp_path):
    """If .codebase-rag/ exists and .omni-rag/ doesn't, rename it."""
    old_dir = tmp_path / ".codebase-rag"
    old_dir.mkdir()
    (old_dir / "test.txt").write_text("data")

    migrate_data_directory(str(tmp_path))

    assert not old_dir.exists()
    assert (tmp_path / ".omni-rag" / "test.txt").exists()


def test_migrate_data_dir_renames_rag_mcp(tmp_path):
    """If .rag-mcp/ exists and .omni-rag/ doesn't, rename it."""
    old_dir = tmp_path / ".rag-mcp"
    old_dir.mkdir()
    (old_dir / "test.txt").write_text("data")

    migrate_data_directory(str(tmp_path))

    assert not old_dir.exists()
    assert (tmp_path / ".omni-rag" / "test.txt").exists()


def test_migrate_data_dir_noop_when_new_exists(tmp_path):
    """If .omni-rag/ already exists, do nothing (don't overwrite)."""
    (tmp_path / ".omni-rag").mkdir()
    (tmp_path / ".codebase-rag").mkdir()

    migrate_data_directory(str(tmp_path))

    assert (tmp_path / ".codebase-rag").exists()  # old untouched
    assert (tmp_path / ".omni-rag").exists()


def test_migrate_data_dir_noop_when_no_old(tmp_path):
    """If no old directories exist, do nothing."""
    migrate_data_directory(str(tmp_path))

    assert not (tmp_path / ".omni-rag").exists()


def test_migrate_prefers_rag_mcp_over_codebase_rag(tmp_path):
    """If both .rag-mcp/ and .codebase-rag/ exist, prefer .rag-mcp/."""
    (tmp_path / ".rag-mcp").mkdir()
    (tmp_path / ".rag-mcp" / "newer.txt").write_text("newer")
    (tmp_path / ".codebase-rag").mkdir()
    (tmp_path / ".codebase-rag" / "older.txt").write_text("older")

    migrate_data_directory(str(tmp_path))

    # .rag-mcp should have been migrated (it's first in the list)
    assert (tmp_path / ".omni-rag" / "newer.txt").exists()
    # .codebase-rag should still exist (untouched)
    assert (tmp_path / ".codebase-rag").exists()
