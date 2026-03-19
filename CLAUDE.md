# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**codebase-rag-mcp** is a Claude Code MCP plugin for token-efficient semantic codebase search. It auto-ingests the current working directory on first search and provides semantic search, codebase overview, structural analysis, and dependency graphs — all cheaper than repeated file reads.

Zero-config by default: local Qdrant storage, ONNX embeddings, no external services required.

## Architecture

- **MCP Server** (`mcp_server/`): FastMCP server with 8 tools
- **Embeddings** (`mcp_server/embeddings/`): Provider abstraction (ONNX, Ollama, OpenAI, Voyage)
- **Analysis** (`mcp_server/analysis/`): Codebase overview and structural analysis
- **Configuration** (`config/`): Centralized pydantic-settings with `RAG_` env prefix

## Key Commands

```bash
# Run the MCP server
codebase-rag

# Run tests
python -m pytest tests/ -v

# Health check
python scripts/health_check.py

# Setup (auto-register with Claude Code)
codebase-rag-setup
```

## Configuration

All settings via environment variables with `RAG_` prefix (see `config/.env.example`):
- `RAG_EMBEDDING_PROVIDER` — onnx (default), ollama, openai, voyage
- `RAG_QDRANT_MODE` — local (default, zero-config) or remote
- `RAG_INGESTION_TIMEOUT_HOURS` — Ingestion timeout (default: 24)
- `RAG_WORKING_DIRECTORY` — Override working directory (default: cwd)

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_codebase` | Semantic search (auto-ingests if needed) |
| `search_codebase_by_file` | Search filtered by file path pattern |
| `get_codebase_context` | Compressed project overview |
| `get_file_signatures` | Function/class signatures |
| `get_dependency_graph` | Internal import/dependency graph |
| `collection_stats` | Index statistics |
| `ingest_current_directory` | Manual re-index (incremental by default) |
| `check_index_status` | Index status + git change detection |
