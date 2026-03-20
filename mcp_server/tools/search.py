"""MCP tools for semantic codebase search."""

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
    """Semantic search over the indexed codebase — use for conceptual questions.

    Use this tool when you need to understand WHERE something is implemented
    or HOW a concept works across the codebase. Much more token-efficient than
    reading files one by one. Auto-ingests on first use.

    Good queries: "authentication middleware", "database connection pooling",
    "error retry logic", "user input validation".

    Do NOT use for: reading a specific known file (use Read), finding files
    by name (use Glob), or literal text search (use Grep).

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


def search_codebase_by_file(query: str, file_pattern: str, n_results: int = 0) -> str:
    """Semantic search restricted to files matching a path pattern.

    Use when you know the general area of the codebase to search in.
    Combines semantic understanding with file path filtering.

    Examples:
        - query="dependency injection", file_pattern="di/module"
        - query="API routes", file_pattern=".py"
        - query="build config", file_pattern=".gradle"

    Args:
        query: What to search for semantically.
        file_pattern: Substring to match in file paths (case-insensitive).
                      E.g. "viewmodel", "shared/src", ".gradle", "test".
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
        hits = search(embedding, n_results, directory_filter=directory, file_pattern=file_pattern)

        if not hits:
            return f"No results matching file pattern '{file_pattern}' for query: \"{query}\""

        return _format_results(f"{query} [files: *{file_pattern}*]", hits)

    except Exception as exc:
        return f"Error searching codebase: {exc}"
