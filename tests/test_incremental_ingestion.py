"""Tests for incremental ingestion and git-based change detection."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mcp_server.ingestion import (
    get_changed_files,
    _get_current_commit,
    _get_last_indexed_commit,
    _save_indexed_commit,
)


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with some files."""
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
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
