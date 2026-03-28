"""Tests for incremental ingestion and git-based change detection."""

import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from mcp_server.ingestion import (
    _get_current_commit,
    _get_last_indexed_commit,
    _save_indexed_commit,
    get_changed_files,
    ingest_incremental,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with some files."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)

    (tmp_path / "main.py").write_text("def hello(): pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)

    return tmp_path


def test_get_current_commit(git_repo):
    """Should return a valid commit hash."""
    commit = _get_current_commit(str(git_repo))
    assert commit is not None
    assert len(commit) == 40  # Full SHA


def test_get_current_commit_non_git(tmp_path):
    """Should return None for non-git directories."""
    commit = _get_current_commit(str(tmp_path))
    assert commit is None


def test_save_and_load_indexed_commit(tmp_path):
    """Should save and load commit hash."""
    _save_indexed_commit(str(tmp_path), "abc123")
    loaded = _get_last_indexed_commit(str(tmp_path))
    assert loaded == "abc123"


def test_load_indexed_commit_missing(tmp_path):
    """Should return None when no marker exists."""
    loaded = _get_last_indexed_commit(str(tmp_path))
    assert loaded is None


def test_get_changed_files_with_changes(git_repo):
    """Should detect new uncommitted files."""
    # Save a commit marker at current HEAD
    commit = _get_current_commit(str(git_repo))
    _save_indexed_commit(str(git_repo), commit)

    # Create a new file (uncommitted)
    (git_repo / "new_file.py").write_text("x = 1\n")

    changed = get_changed_files(str(git_repo))
    assert "new_file.py" in changed


def test_get_changed_files_no_changes(git_repo):
    """Should return empty when nothing changed (excluding .codebase-rag marker)."""
    # Add .codebase-rag to gitignore so it doesn't show as untracked
    (git_repo / ".gitignore").write_text(".codebase-rag/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=git_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add gitignore"], cwd=git_repo, capture_output=True)

    commit = _get_current_commit(str(git_repo))
    _save_indexed_commit(str(git_repo), commit)

    changed = get_changed_files(str(git_repo))
    assert len(changed) == 0


def test_get_changed_files_committed_changes(git_repo):
    """Should detect files changed between commits."""
    # Save marker at initial commit
    commit = _get_current_commit(str(git_repo))
    _save_indexed_commit(str(git_repo), commit)

    # Make a new commit
    (git_repo / "main.py").write_text("def hello(): return 'hi'\n")
    subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=git_repo, capture_output=True)

    changed = get_changed_files(str(git_repo))
    assert "main.py" in changed


# ---------------------------------------------------------------------------
# Tests for deleted-file handling in ingest_incremental
# ---------------------------------------------------------------------------


def _make_git_repo_with_marker(tmp_path):
    """Helper: init a git repo, commit a file, and save the ingestion marker."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "main.py").write_text("def hello(): pass\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, capture_output=True)
    commit = _get_current_commit(str(tmp_path))
    _save_indexed_commit(str(tmp_path), commit)
    return tmp_path


@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.get_changed_files")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=4)
def test_ingest_incremental_deleted_file_calls_delete(
    mock_dim,
    mock_ensure,
    mock_changed,
    mock_delete_file,
    tmp_path,
):
    """ingest_incremental should call delete_file_points for files that no longer exist."""
    # Set up repo with a marker so incremental path is taken
    _make_git_repo_with_marker(tmp_path)

    # Report a file as changed that does NOT exist on disk (i.e. it was deleted)
    mock_changed.return_value = ["deleted_module.py"]

    result = ingest_incremental(str(tmp_path))

    mock_delete_file.assert_called_once_with("deleted_module.py", str(tmp_path))
    # No files to ingest — should return the "no indexable changes" message
    assert "No indexable changes" in result


@patch("mcp_server.ingestion.upsert_chunks", return_value=3)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.get_changed_files")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=4)
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "1", "text": "x", "file_path": "main.py", "directory": "", "chunk_index": 0, "ingested_at": ""}],
        [[0.1, 0.2, 0.3, 0.4]],
        ["main.py"],
    ),
)
def test_ingest_incremental_changed_file_deletes_before_upsert(
    mock_embed,
    mock_dim,
    mock_ensure,
    mock_changed,
    mock_delete_file,
    mock_upsert,
    tmp_path,
):
    """ingest_incremental should delete old chunks before upserting new ones for changed files."""
    _make_git_repo_with_marker(tmp_path)

    # main.py exists and is reported as changed
    (tmp_path / "main.py").write_text("def hello(): return 'updated'\n")
    mock_changed.return_value = ["main.py"]

    result = ingest_incremental(str(tmp_path))

    # delete_file_points must have been called for main.py
    mock_delete_file.assert_called_once_with("main.py", str(tmp_path))
    # upsert must also have been called
    mock_upsert.assert_called_once()
    assert "Incrementally indexed" in result


@patch("mcp_server.ingestion.upsert_chunks", return_value=2)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.get_changed_files")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=4)
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [
            {"id": "1", "text": "a", "file_path": "a.py", "directory": "", "chunk_index": 0, "ingested_at": ""},
            {"id": "2", "text": "b", "file_path": "b.py", "directory": "", "chunk_index": 0, "ingested_at": ""},
        ],
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        ["a.py", "b.py"],
    ),
)
def test_ingest_incremental_mixed_deleted_and_changed(
    mock_embed,
    mock_dim,
    mock_ensure,
    mock_changed,
    mock_delete_file,
    mock_upsert,
    tmp_path,
):
    """ingest_incremental handles a mix of deleted and changed files correctly."""
    _make_git_repo_with_marker(tmp_path)

    # a.py and b.py exist (changed), gone.py does not (deleted)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    mock_changed.return_value = ["a.py", "b.py", "gone.py"]

    ingest_incremental(str(tmp_path))

    # delete_file_points called for deleted file and both changed files
    actual_args = sorted(c.args[0] for c in mock_delete_file.call_args_list)
    assert actual_args == ["a.py", "b.py", "gone.py"]
    # All calls use the same directory
    for c in mock_delete_file.call_args_list:
        assert c.args[1] == str(tmp_path)
