"""Data directory and collection migration from codebase-rag to rag-mcp."""

import logging
import os
from pathlib import Path

log = logging.getLogger("rag-mcp")


def migrate_data_directory(directory: str) -> None:
    """Rename .codebase-rag/ to .rag-mcp/ if the old directory exists.

    On Windows, file locks may prevent renaming if the Qdrant client is open.
    In that case, fall back to using the old path with a deprecation warning.
    """
    old = Path(directory) / ".codebase-rag"
    new = Path(directory) / ".rag-mcp"

    if not old.exists() or new.exists():
        return

    try:
        os.rename(str(old), str(new))
        log.info("Migrated data directory: %s -> %s", old, new)
    except OSError as e:
        log.warning(
            "Could not rename %s to %s: %s. Using old path with deprecation warning.",
            old,
            new,
            e,
        )
