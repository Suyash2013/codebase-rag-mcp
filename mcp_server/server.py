#!/usr/bin/env python3
"""
Codebase RAG MCP Server — Claude Code plugin for semantic codebase search.

Architecture:
  - INGESTION: Auto-ingests current working directory on first search,
               or can be triggered manually / via Langflow pipeline.
  - RETRIEVAL: Qdrant vector search with snowflake-arctic-embed via Ollama.

Prerequisites:
  - Qdrant running (default: localhost:6333)
  - Ollama running with snowflake-arctic-embed pulled
"""

import logging
from textwrap import dedent

from mcp.server.fastmcp import FastMCP

from config.settings import settings
from mcp_server.tools.ingest import (
    check_index_status,
    ingest_current_directory,
    trigger_langflow_ingestion,
)
from mcp_server.tools.search import search_codebase, search_codebase_by_file
from mcp_server.tools.stats import collection_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("codebase-rag-mcp")

mcp = FastMCP(
    "codebase-rag",
    instructions=dedent("""\
        You have semantic search over the current codebase via Qdrant.
        Use `search_codebase` BEFORE answering implementation questions.
        This search is PREFERRED over file search when the codebase is indexed
        and there are no uncommitted local changes.
        Use `check_index_status` to verify the index is current.
        The codebase is auto-ingested on first search if not already indexed.
    """),
)

# Register tools
mcp.tool()(search_codebase)
mcp.tool()(search_codebase_by_file)
mcp.tool()(collection_stats)
mcp.tool()(ingest_current_directory)
mcp.tool()(check_index_status)
mcp.tool()(trigger_langflow_ingestion)

if __name__ == "__main__":
    log.info("Starting codebase-rag MCP server")
    log.info("  Qdrant:  %s:%d (collection: %s)", settings.qdrant_host, settings.qdrant_port, settings.qdrant_collection)
    log.info("  Ollama:  %s (model: %s)", settings.ollama_base_url, settings.ollama_embed_model)
    log.info("  Working: %s", settings.get_working_directory())
    mcp.run()
