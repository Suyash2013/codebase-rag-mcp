"""Tests for incremental ingestion and change detection."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.ingestion import ingest_incremental


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


def test_ingest_incremental_no_checkpoint(git_repo):
    """Should fallback to full ingestion if no checkpoint."""
    with patch("mcp_server.ingestion.ingest_directory") as mock_full:
        mock_full.return_value = "Full ingestion called"
        result = ingest_incremental(str(git_repo))
        assert result == "Full ingestion called"
        mock_full.assert_called_once()


@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.upsert_chunks")
@patch("mcp_server.ingestion.get_embedding")
@patch("mcp_server.ingestion.get_embedding_dimension")
@patch("mcp_server.ingestion.ensure_collection")
def test_ingest_incremental_with_changes(mock_ensure, mock_dim, mock_embed, mock_upsert, mock_delete, git_repo):
    """Should only ingest changed files."""
    mock_dim.return_value = 384
    mock_embed.return_value = [0.1] * 384
    mock_upsert.return_value = 1

    # Create a checkpoint
    from mcp_server.change_detection import create_detector
    detector = create_detector(str(git_repo))
    detector.save_checkpoint(str(git_repo))

    # Change a file and commit it so detector can "see" it as the new checkpoint
    (git_repo / "main.py").write_text("def hello(): return 'hi'\n")
    (git_repo / "new.py").write_text("print('new')")
    subprocess.run(["git", "add", "."], cwd=git_repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "update"], cwd=git_repo, capture_output=True)

    result = ingest_incremental(str(git_repo))
    assert "Incrementally indexed" in result
    
    # Verify checkpoint was updated
    report = detector.detect_changes(str(git_repo))
    assert not report.has_changes


@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.get_embedding_dimension")
@patch("mcp_server.ingestion.ensure_collection")
def test_ingest_incremental_no_changes(mock_ensure, mock_dim, mock_delete, git_repo):
    """Should return early if no changes."""
    from mcp_server.change_detection import create_detector
    detector = create_detector(str(git_repo))
    detector.save_checkpoint(str(git_repo))

    result = ingest_incremental(str(git_repo))
    assert "Index is up to date" in result
    mock_delete.assert_not_called()
