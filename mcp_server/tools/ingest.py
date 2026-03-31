"""MCP tools for ingestion control."""

import json

from config.settings import settings
from mcp_server.ingestion import (
    check_local_changes,
    ingest_directory,
    ingest_incremental,
    needs_ingestion,
)


def ingest(
    force: bool = False,
    include_extensions: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
) -> str:
    """Manually trigger indexing of the current working directory.

    Auto-indexes on first search, so you rarely need this.
    By default, does incremental indexing (only changed files).
    Use force=True to do a full re-index from scratch.

    Args:
        force: If True, full re-index. If False, incremental update.
        include_extensions: Optional list of file extensions to include
            (e.g. [".py", ".ts", "js"]). When provided, ONLY files with these
            extensions are indexed; the default TEXT_EXTENSIONS allowlist is
            replaced. Leading dots are optional.
        exclude_extensions: Optional list of file extensions to exclude
            (e.g. [".txt", ".md"]). Files with these extensions are skipped
            even if they would otherwise be indexed. Leading dots are optional.
    """
    directory = settings.get_working_directory()

    try:
        if force:
            return ingest_directory(directory, include_extensions, exclude_extensions)

        if needs_ingestion(directory):
            return ingest_directory(directory, include_extensions, exclude_extensions)

        return ingest_incremental(directory, include_extensions, exclude_extensions)

    except Exception as exc:
        return f"Error during ingestion: {exc}"


def check_status() -> str:
    """Check whether the index for the current directory is up to date.

    Call this FIRST if you're unsure whether to use semantic search or
    fall back to file-by-file reading. Returns whether the directory is
    indexed and whether there are local changes that might make the
    index stale.
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
