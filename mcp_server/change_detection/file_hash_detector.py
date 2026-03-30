"""File modification time + hash based change detection for non-git directories."""

import hashlib
import json
import logging
import os
from pathlib import Path
from mcp_server.change_detection.base import ChangeDetector, ChangeReport

log = logging.getLogger("rag-mcp")

DATA_DIR = ".rag-mcp"
MANIFEST_FILE = "file_manifest.json"


class FileHashDetector(ChangeDetector):
    """File modification time + hash based change detection for non-git directories."""

    def detect_changes(self, directory: str) -> ChangeReport:
        manifest_path = Path(directory) / DATA_DIR / MANIFEST_FILE
        if not manifest_path.exists():
            return ChangeReport(has_changes=True, details="No prior manifest found")

        old_manifest = json.loads(manifest_path.read_text())
        current_manifest = self._build_manifest(directory)

        changed = []
        deleted = []

        # Check for modified or new files
        for rel_path, info in current_manifest.items():
            if rel_path not in old_manifest:
                changed.append(rel_path)
            elif info["hash"] != old_manifest[rel_path]["hash"]:
                changed.append(rel_path)

        # Check for deleted files
        for rel_path in old_manifest:
            if rel_path not in current_manifest:
                deleted.append(rel_path)

        has_changes = bool(changed or deleted)
        return ChangeReport(
            has_changes=has_changes,
            changed_files=changed,
            deleted_files=deleted,
            details=f"{len(changed)} changed, {len(deleted)} deleted"
        )

    def save_checkpoint(self, directory: str) -> None:
        manifest = self._build_manifest(directory)
        manifest_path = Path(directory) / DATA_DIR / MANIFEST_FILE
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def has_checkpoint(self, directory: str) -> bool:
        return (Path(directory) / DATA_DIR / MANIFEST_FILE).exists()

    def _build_manifest(self, directory: str) -> dict:
        manifest = {}
        dir_path = Path(directory)
        for root, dirs, files in os.walk(directory):
            # Skip hidden and data dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    rel = str(fpath.relative_to(dir_path)).replace("\\", "/")
                    stat = fpath.stat()
                    file_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
                    manifest[rel] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "hash": file_hash,
                    }
                except (OSError, PermissionError):
                    continue
        return manifest
