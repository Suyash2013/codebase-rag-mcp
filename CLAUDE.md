# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**omni-rag-mcp** is a general-purpose RAG MCP plugin for token-efficient semantic search over any directory of files. It auto-ingests the current working directory on first search and provides hybrid search (BM25 + semantic), directory overview, structural analysis, and dependency graphs — all cheaper than repeated file reads.

Zero-config by default: local Qdrant storage, ONNX embeddings, no external services required. Supports code, markdown, PDFs, CSVs, and more via pluggable extractors.

## Architecture

- **MCP Server** (`mcp_server/`): FastMCP server with 8 core tools + deprecated aliases
- **Extractors** (`mcp_server/extractors/`): Plugin system for content extraction (code, markdown, structured data, PDF, DOCX, image)
- **Chunkers** (`mcp_server/chunkers/`): Format-aware chunking (code boundaries, markdown headings, CSV rows, etc.)
- **Storage** (`mcp_server/storage/`): BM25 index + hybrid search rank fusion
- **Embeddings** (`mcp_server/embeddings/`): Provider abstraction (ONNX, Ollama, OpenAI, Voyage)
- **Change Detection** (`mcp_server/change_detection/`): Git-based or file-hash-based incremental indexing
- **Analysis** (`mcp_server/analysis/`): Directory overview and structural analysis
- **Configuration** (`config/`): Centralized pydantic-settings with `OMNI_RAG_` env prefix

## Key Commands

```bash
# Run the MCP server
omni-rag

# Run tests
python -m pytest tests/ -v

# Health check
python scripts/health_check.py

# Setup (auto-register with Claude Code)
omni-rag-setup
```

## Configuration

All settings via environment variables with `OMNI_RAG_` prefix (see `config/.env.example`):
- `OMNI_RAG_EMBEDDING_PROVIDER` — onnx (default), ollama, openai, voyage
- `OMNI_RAG_QDRANT_MODE` — local (default, zero-config) or remote
- `OMNI_RAG_INGESTION_TIMEOUT_HOURS` — Ingestion timeout (default: 24)
- `OMNI_RAG_WORKING_DIRECTORY` — Override working directory (default: cwd)
- `OMNI_RAG_HYBRID_SEARCH_ENABLED` — Enable BM25+semantic hybrid search (default: true)
- `OMNI_RAG_ENABLE_CODE_PLUGIN` — Enable code-specific tools (default: true)

Legacy `RAG_` prefix env vars are still supported with deprecation warnings.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search` | Hybrid search over indexed files (auto-ingests if needed) |
| `search_by_file` | Search filtered by file path pattern |
| `get_context` | Compressed directory overview |
| `get_file_signatures` | Function/class signatures (code plugin) |
| `get_dependency_graph` | Internal import/dependency graph (code plugin) |
| `stats` | Index statistics |
| `ingest` | Manual re-index (incremental by default) |
| `check_status` | Index status + change detection |

Old tool names (`search_codebase`, `get_codebase_context`, etc.) are registered as deprecated aliases.
