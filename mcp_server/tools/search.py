"""MCP tools for semantic codebase search."""

import json

from config.settings import settings
from mcp_server.embeddings import get_embedding
from mcp_server.ingestion import ingest_directory, needs_ingestion
from mcp_server.qdrant_client import search


def _format_results(query: str, hits: list[dict]) -> str:
    """Format search results into a readable string for Claude Code."""
    if not hits:
        return f"No results found for: {query}"

    parts = [f'Found {len(hits)} results for: "{query}"\n']

    for i, hit in enumerate(hits):
        parts.append(
            f"--- Result {i + 1} (score: {hit['score']:.4f}) ---\n"
            f"Source: {hit['file_path']}\n"
            f"```\n{hit['text']}\n```"
        )

    return "\n\n".join(parts)


def search_codebase(query: str, n_results: int = 0) -> str:
    """Semantic search over the indexed codebase.

    AUTO-INGESTS the current working directory if not yet indexed.
    Use this tool instead of file search when the codebase has been
    indexed and there are no uncommitted local changes.

    Args:
        query: Natural language description of what you're looking for.
               Be specific — "authentication middleware error handling"
               works better than just "auth".
        n_results: Number of results to return (1-20, default 10).
    """
    if n_results <= 0:
        n_results = settings.default_n_results
    n_results = max(1, min(n_results, settings.max_n_results))

    directory = settings.get_working_directory()

    try:
        if needs_ingestion(directory):
            ingest_directory(directory)

        embedding = get_embedding(query)
        hits = search(embedding, n_results, directory_filter=directory)
        return _format_results(query, hits)

    except Exception as exc:
        return f"Error searching codebase: {exc}"


def search_codebase_by_file(
    query: str, file_pattern: str, n_results: int = 0
) -> str:
    """Search the codebase with a file path filter.

    Like search_codebase, but restricts results to files whose path
    contains the given pattern (case-insensitive substring match).

    Args:
        query: What to search for semantically.
        file_pattern: Substring to match in file paths, e.g. "viewmodel",
                      "shared/src", ".gradle", "di/module".
        n_results: Number of results to return (1-20, default 10).
    """
    if n_results <= 0:
        n_results = settings.default_n_results
    n_results = max(1, min(n_results, settings.max_n_results))

    directory = settings.get_working_directory()

    try:
        if needs_ingestion(directory):
            ingest_directory(directory)

        embedding = get_embedding(query)
        hits = search(
            embedding, n_results, directory_filter=directory, file_pattern=file_pattern
        )

        if not hits:
            return (
                f"No results matching file pattern '{file_pattern}' "
                f'for query: "{query}"'
            )

        return _format_results(f"{query} [files: *{file_pattern}*]", hits)

    except Exception as exc:
        return f"Error searching codebase: {exc}"
