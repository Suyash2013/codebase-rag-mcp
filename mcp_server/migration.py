"""Data directory and collection migration to omni-rag."""

import logging
import os
from pathlib import Path

log = logging.getLogger("omni-rag")


def migrate_data_directory(directory: str) -> None:
    """Migrate old data directories to .omni-rag/.

    Handles:
    - .codebase-rag/ -> .omni-rag/
    - .rag-mcp/ -> .omni-rag/

    On Windows, file locks may prevent renaming if the Qdrant client is open.
    In that case, fall back to using the old path with a deprecation warning.
    """
    new = Path(directory) / ".omni-rag"
    if new.exists():
        return

    # Try migrating from old directories (in order of preference)
    for old_name in [".rag-mcp", ".codebase-rag"]:
        old = Path(directory) / old_name
        if old.exists():
            try:
                os.rename(str(old), str(new))
                log.info("Migrated data directory: %s -> %s", old, new)
                return
            except OSError as e:
                log.warning(
                    "Could not rename %s to %s: %s. Using old path.",
                    old,
                    new,
                    e,
                )
                return


def detect_collection_name() -> str:
    """Auto-detect the Qdrant collection name for backward compatibility.

    Tries the new default first, then falls back to old names.
    Returns the collection name to use.
    """
    from config.settings import settings
    from mcp_server.qdrant_client import get_client

    configured = settings.qdrant_collection
    try:
        client = get_client()
        collections = [c.name for c in client.get_collections().collections]

        # If the configured collection exists, use it
        if configured in collections:
            return configured

        # Fall back to old collection names
        for old_name in ["documents", "codebase"]:
            if old_name in collections and old_name != configured:
                log.info("Using existing collection '%s' (configured: '%s')", old_name, configured)
                return old_name

    except Exception:
        pass

    return configured
