#!/usr/bin/env python3
"""
Omni-RAG MCP Server — plugin for semantic search over any directory of files.

Architecture:
  - INGESTION: Auto-ingests current working directory on first search,
               or can be triggered manually.
  - RETRIEVAL: Hybrid search (BM25 + semantic) with configurable embedding providers.
"""

import logging
from textwrap import dedent

from mcp.server.fastmcp import FastMCP

from config.settings import settings
from mcp_server.migration import migrate_data_directory
from mcp_server.tools.context import get_context
from mcp_server.tools.ingest import (
    check_status,
    ingest,
)
from mcp_server.tools.search import search, search_by_file
from mcp_server.tools.stats import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("omni-rag")

mcp = FastMCP(
    "omni-rag",
    instructions=dedent("""\
        You have semantic search via the omni-rag plugin.

        WHEN TO USE THESE TOOLS:
        - Use `search` for conceptual/semantic questions: "where is
          authentication handled", "how does the retry logic work", "find the
          database connection setup". This is PREFERRED over file-by-file reading
          when the directory is indexed — it saves significant tokens.
        - Use `search_by_file` when you know the general area: search
          within "viewmodel" files, ".gradle" files, "test" files, etc.
        - Use `check_status` FIRST if you're unsure whether the index
          is current or if files have changed since last indexing.
        - Use `get_context` FIRST when starting on an unfamiliar
          directory — returns file type breakdown, structure, and dependencies
          in one cheap call.
        - Use `get_file_signatures` to understand a module's API surface
          (functions, classes, params) without reading every file.
        - Use `get_dependency_graph` to understand which files import from
          which — helps find the right place to make changes.
        - Use `stats` to check index size and configuration.
        - Use `ingest` with force=True after significant
          changes to refresh the index.

        WHEN NOT TO USE:
        - For reading a specific known file path (use Read/cat directly).
        - For finding files by exact name or glob pattern (use Glob/find).
        - For simple literal text search (use Grep/rg).

        The directory auto-indexes on first search. No setup required.
    """),
)

# Register core tools
mcp.tool()(search)
mcp.tool()(search_by_file)
mcp.tool()(stats)
mcp.tool()(ingest)
mcp.tool()(check_status)
mcp.tool()(get_context)

# Register code-specific tools (conditionally)
if settings.enable_code_plugin:
    from mcp_server.tools.structure import get_dependency_graph, get_file_signatures

    mcp.tool()(get_file_signatures)
    mcp.tool()(get_dependency_graph)


# Deprecated aliases — thin wrappers for backward compatibility
@mcp.tool()
def search_codebase(query: str, n_results: int | None = None) -> str:
    """[Deprecated: use 'search'] Semantic search over indexed files."""
    log.warning("Tool 'search_codebase' is deprecated, use 'search' instead")
    return search(query, n_results)


@mcp.tool()
def search_codebase_by_file(query: str, file_pattern: str, n_results: int | None = None) -> str:
    """[Deprecated: use 'search_by_file'] Search filtered by file pattern."""
    log.warning("Tool 'search_codebase_by_file' is deprecated, use 'search_by_file' instead")
    return search_by_file(query, file_pattern, n_results)


@mcp.tool()
def get_codebase_context() -> str:
    """[Deprecated: use 'get_context'] Get directory overview."""
    log.warning("Tool 'get_codebase_context' is deprecated, use 'get_context' instead")
    return get_context()


@mcp.tool()
def collection_stats() -> str:
    """[Deprecated: use 'stats'] Get index statistics."""
    log.warning("Tool 'collection_stats' is deprecated, use 'stats' instead")
    return stats()


@mcp.tool()
def ingest_current_directory(
    force: bool = False,
    include_extensions: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
) -> str:
    """[Deprecated: use 'ingest'] Manually trigger indexing."""
    log.warning("Tool 'ingest_current_directory' is deprecated, use 'ingest' instead")
    return ingest(force, include_extensions, exclude_extensions)


@mcp.tool()
def check_index_status() -> str:
    """[Deprecated: use 'check_status'] Check index freshness."""
    log.warning("Tool 'check_index_status' is deprecated, use 'check_status' instead")
    return check_status()


if __name__ == "__main__":
    log.info("Starting omni-rag server")
    migrate_data_directory(settings.get_working_directory())
    log.info("  Embedding: %s", settings.embedding_provider)
    log.info(
        "  Qdrant:    %s",
        (
            f"local ({settings.get_qdrant_local_path()})"
            if settings.qdrant_mode == "local"
            else f"remote ({settings.qdrant_host}:{settings.qdrant_port})"
        ),
    )
    log.info("  Working:   %s", settings.get_working_directory())
    mcp.run()
