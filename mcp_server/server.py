#!/usr/bin/env python3
"""
Codebase RAG MCP Server — Claude Code plugin for semantic codebase search.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("codebase-rag-mcp")

mcp = FastMCP(
    "codebase-rag",
    instructions=dedent("""\
        You have semantic codebase search via the codebase-rag plugin.

        WHEN TO USE THESE TOOLS:
        - Use `search_codebase` for conceptual/semantic questions: "where is
          authentication handled", "how does the retry logic work", "find the
          database connection setup". This is PREFERRED over file-by-file reading
          when the codebase is indexed — it saves significant tokens.
        - Use `search_codebase_by_file` when you know the general area: search
          within "viewmodel" files, ".gradle" files, "test" files, etc.
        - Use `check_index_status` FIRST if you're unsure whether the index
          is current or if the codebase has changed since last indexing.
        - Use `get_codebase_context` FIRST when starting on an unfamiliar
          codebase — returns language breakdown, structure, and dependencies
          in one cheap call.
        - Use `get_file_signatures` to understand a module's API surface
          (functions, classes, params) without reading every file.
        - Use `get_dependency_graph` to understand which files import from
          which — helps find the right place to make changes.
        - Use `collection_stats` to check index size and configuration.
        - Use `ingest_current_directory` with force=True after significant
          code changes to refresh the index.

        WHEN NOT TO USE:
        - For reading a specific known file path (use Read/cat directly).
        - For finding files by exact name or glob pattern (use Glob/find).
        - For simple literal text search (use Grep/rg).

        The codebase auto-indexes on first search. No setup required.
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
    log.info("Starting codebase-rag MCP server")
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
