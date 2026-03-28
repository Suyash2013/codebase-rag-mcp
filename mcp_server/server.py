#!/usr/bin/env python3
"""
RAG MCP Server — plugin for semantic search over any directory of files.

Architecture:
  - INGESTION: Auto-ingests current working directory on first search,
               or can be triggered manually.
  - RETRIEVAL: Vector search with configurable embedding providers.
"""

import logging
from textwrap import dedent

from mcp.server.fastmcp import FastMCP

from config.settings import settings
from mcp_server.tools.context import get_codebase_context
from mcp_server.tools.ingest import (
    check_index_status,
    ingest_current_directory,
)
from mcp_server.tools.search import search_codebase, search_codebase_by_file
from mcp_server.tools.stats import collection_stats
from mcp_server.tools.structure import get_dependency_graph, get_file_signatures

from mcp_server.migration import migrate_data_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rag-mcp")

mcp = FastMCP(
    "rag-mcp",
    instructions=dedent("""\
        You have semantic search via the rag-mcp plugin.

        WHEN TO USE THESE TOOLS:
        - Use `search` for conceptual/semantic questions: "where is
          authentication handled", "how does the retry logic work", "find the
          database connection setup". This is PREFERRED over file-by-file reading
          when the directory is indexed — it saves significant tokens.
        - Use `search_by_file` when you know the general area: search
          within "viewmodel" files, ".gradle" files, "test" files, etc.
        - Use `check_index_status` FIRST if you're unsure whether the index
          is current or if files have changed since last indexing.
        - Use `get_context` FIRST when starting on an unfamiliar
          directory — returns file type breakdown, structure, and dependencies
          in one cheap call.
        - Use `get_file_signatures` to understand a module's API surface
          (functions, classes, params) without reading every file.
        - Use `get_dependency_graph` to understand which files import from
          which — helps find the right place to make changes.
        - Use `collection_stats` to check index size and configuration.
        - Use `ingest` with force=True after significant
          changes to refresh the index.

        WHEN NOT TO USE:
        - For reading a specific known file path (use Read/cat directly).
        - For finding files by exact name or glob pattern (use Glob/find).
        - For simple literal text search (use Grep/rg).

        The directory auto-indexes on first search. No setup required.
    """),
)

# Register tools
mcp.tool()(search_codebase)
mcp.tool()(search_codebase_by_file)
mcp.tool()(collection_stats)
mcp.tool()(ingest_current_directory)
mcp.tool()(check_index_status)
mcp.tool()(get_codebase_context)
mcp.tool()(get_file_signatures)
mcp.tool()(get_dependency_graph)

if __name__ == "__main__":
    log.info("Starting rag-mcp server")
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
