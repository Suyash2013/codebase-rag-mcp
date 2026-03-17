# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**codebase-rag-mcp** is a Claude Code MCP plugin that provides semantic codebase search via Qdrant and Ollama (snowflake-arctic-embed). It auto-ingests the current working directory on first search and serves as the preferred search method when no local changes exist.

The project also includes Langflow agent pipeline configurations and custom components that integrate bidirectionally with the MCP server.

## Architecture

- **MCP Server** (`mcp_server/`): FastMCP server with tools for search, stats, and ingestion
- **Langflow Flows** (`flows/`): DAG workflow definitions for agent pipelines
- **Custom Components** (`components/`): Langflow components including an MCP bridge
- **Configuration** (`config/`): Centralized pydantic-settings with `RAG_` env prefix

## Key Commands

```bash
# Run the MCP server
uv run mcp_server/server.py

# Run tests
pytest tests/ -v

# Health check (verify Ollama + Qdrant)
python scripts/health_check.py

# Validate flow JSON files
python scripts/validate_flows.py

# Validate a specific JSON file
python -m json.tool flows/simple_agent.json > /dev/null 2>&1 && echo "Valid" || echo "Invalid"
```

## Configuration

All settings via environment variables with `RAG_` prefix (see `config/.env.example`):
- `RAG_QDRANT_HOST` / `RAG_QDRANT_PORT` — Qdrant connection
- `RAG_OLLAMA_EMBED_MODEL` — Embedding model (default: snowflake-arctic-embed:latest)
- `RAG_INGESTION_TIMEOUT_HOURS` — Ingestion timeout (default: 24)
- `RAG_WORKING_DIRECTORY` — Override working directory (default: cwd)

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_codebase` | Semantic search (auto-ingests if needed) |
| `search_codebase_by_file` | Search filtered by file path pattern |
| `collection_stats` | Qdrant collection statistics |
| `ingest_current_directory` | Manual ingestion trigger |
| `check_index_status` | Index status + git change detection |
| `trigger_langflow_ingestion` | Trigger Langflow pipeline via REST |

## When Modifying Flow JSON

- Validate with `python -m json.tool` before considering work complete
- Edge `id`, `sourceHandle`, `targetHandle` fields contain JSON-as-strings requiring proper escaping
- Always use UTF-8 encoding
