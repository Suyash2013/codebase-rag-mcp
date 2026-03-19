"""MCP tools for ingestion control."""

import json

from config.settings import settings
from mcp_server.ingestion import (
    check_local_changes,
    ingest_directory,
    ingest_incremental,
    needs_ingestion,
)


def ingest_current_directory(force: bool = False) -> str:
    """Manually trigger indexing of the current working directory.

    The codebase auto-indexes on first search, so you rarely need this.
    By default, does incremental indexing (only changed files).
    Use force=True to do a full re-index from scratch.

    Args:
        force: If True, full re-index. If False, incremental update.
    """
    directory = settings.get_working_directory()

    try:
        if force:
            return ingest_directory(directory)

        if needs_ingestion(directory):
            return ingest_directory(directory)

        return ingest_incremental(directory)

    except Exception as exc:
        return f"Error during ingestion: {exc}"


def check_index_status() -> str:
    """Check whether the codebase index is current.

    Call this FIRST if you're unsure whether to use semantic search or
    fall back to file-by-file reading. Returns whether the directory is
    indexed and whether there are uncommitted changes that might make
    the index stale.
    """
    directory = settings.get_working_directory()

    try:
        indexed = not needs_ingestion(directory)
        changes = check_local_changes(directory)

        result = {
            "directory": directory,
            "is_indexed": indexed,
            "recommendation": (
                "Use semantic search"
                if indexed and not changes.get("has_changes")
                else "Consider re-indexing or use file search for changed files"
                if indexed and changes.get("has_changes")
                else "Ingestion needed — will auto-index on first search"
            ),
            **changes,
        }

        return json.dumps(result, indent=2)

    except Exception as exc:
        return f"Error checking index status: {exc}"
