"""Change detection factory and re-exports."""

from pathlib import Path

from mcp_server.change_detection.base import ChangeDetector, ChangeReport
from mcp_server.change_detection.file_hash_detector import FileHashDetector
from mcp_server.change_detection.git_detector import GitDetector


def create_detector(directory: str) -> ChangeDetector:
    """Factory: returns GitDetector if in a git repo, else FileHashDetector."""
    if (Path(directory) / ".git").exists():
        return GitDetector()
    return FileHashDetector()


__all__ = ["ChangeDetector", "ChangeReport", "FileHashDetector", "GitDetector", "create_detector"]
