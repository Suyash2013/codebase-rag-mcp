"""Git-based change detection."""

import logging
import subprocess
from pathlib import Path

from mcp_server.change_detection.base import ChangeDetector, ChangeReport

log = logging.getLogger("omni-rag")

DATA_DIR = ".omni-rag"


class GitDetector(ChangeDetector):
    """Git-based change detection."""

    def detect_changes(self, directory: str) -> ChangeReport:
        last_commit = self._get_last_indexed_commit(directory)
        if not last_commit:
            return ChangeReport(has_changes=True, details="No prior index found")

        current_commit = self._get_current_commit(directory)
        changed = []
        deleted = []

        # Changes since last indexed commit
        if current_commit and current_commit != last_commit:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", last_commit, current_commit],
                    capture_output=True,
                    text=True,
                    cwd=directory,
                    timeout=30,
                )
                if result.returncode == 0:
                    for f in result.stdout.splitlines():
                        f = f.strip()
                        if f and not self._should_skip(f):
                            if (Path(directory) / f).exists():
                                changed.append(f)
                            else:
                                deleted.append(f)
            except Exception as e:
                log.warning("git diff failed: %s", e)

        # Uncommitted changes
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if len(line) > 3:
                        status = line[:2]
                        filepath = line[3:].strip()

                        if self._should_skip(filepath):
                            continue

                        if status[0] == "D" or status[1] == "D":
                            if filepath not in deleted:
                                deleted.append(filepath)
                        elif filepath not in changed:
                            changed.append(filepath)
        except Exception as e:
            log.warning("git status failed: %s", e)

        has_changes = bool(changed or deleted)
        return ChangeReport(
            has_changes=has_changes,
            changed_files=changed,
            deleted_files=deleted,
            details=f"{len(changed)} changed, {len(deleted)} deleted since {last_commit[:8]}",
        )

    def _should_skip(self, path: str) -> bool:
        """Check if a path should be ignored by change detection."""
        # Ignore the data directory itself
        if path.startswith(DATA_DIR) or f"/{DATA_DIR}" in path:
            return True
        # Ignore other common skip directories if they show up
        skip_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv"}
        parts = Path(path).parts
        return bool(any(p in skip_dirs for p in parts))

    def save_checkpoint(self, directory: str) -> None:
        commit = self._get_current_commit(directory)
        if commit:
            marker = Path(directory) / DATA_DIR / "last_commit.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(commit)

    def has_checkpoint(self, directory: str) -> bool:
        return self._get_last_indexed_commit(directory) is not None

    def _get_current_commit(self, directory: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _get_last_indexed_commit(self, directory: str) -> str | None:
        marker = Path(directory) / DATA_DIR / "last_commit.txt"
        if marker.exists():
            return marker.read_text().strip()
        return None
