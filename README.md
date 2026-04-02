# omni-rag-mcp

[![PyPI](https://img.shields.io/pypi/v/omni-rag-mcp)](https://pypi.org/project/omni-rag-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

General-purpose RAG MCP plugin — index any directory, search with hybrid BM25 + semantic, zero-config.

A token-efficient semantic search plugin for Claude Code (and any MCP client). Auto-ingests your working directory on first search and provides hybrid search, directory overview, structural analysis, and dependency graphs — all cheaper than repeated file reads.

Zero-config by default: local Qdrant storage, ONNX embeddings, no external services required. Supports code, markdown, PDFs, CSVs, and more via pluggable extractors.

## Key Features

- **Zero-config auto-indexing** — auto-ingests the working directory on first search, no manual setup needed
- **Hybrid search** — combines BM25 keyword search (weight 0.3) with semantic vector search (weight 0.7) via reciprocal rank fusion
- **Qdrant vector store** — local on-disk mode (zero-config default) or remote Docker/cloud mode
- **4 embedding providers:**
  - `onnx` (default) — local ONNX Runtime, model: `all-MiniLM-L6-v2`, zero-config, no API keys
  - `ollama` — local Ollama, default model: `snowflake-arctic-embed:latest`
  - `openai` — OpenAI API, default model: `text-embedding-3-small`
  - `voyage` — Voyage AI API, default model: `voyage-code-3`
- **Smart chunking** — 5 chunkers: code-aware, markdown-aware, paragraph, recursive, structured
- **Multi-format extraction** — code files (40+ extensions), markdown, PDF (optional), DOCX (optional), images/OCR (optional), structured data (JSON/YAML/CSV/XML), plain text
- **Change detection** — git-based (if in a git repo) or file-hash-based (fallback), for incremental re-indexing
- **Code analysis plugin** (enabled by default) — `get_file_signatures` (function/class signatures) and `get_dependency_graph` (internal import graph)

## Quick Start

```bash
pip install omni-rag-mcp
omni-rag-setup
```

Restart Claude Code and the plugin auto-indexes your working directory on first search.

## How It Works

```
Your Files  ->  Extractors  ->  Chunking  ->  Embedding  ->  Qdrant (local)
                                                                 |
Claude Code ->  MCP Tool Call  ->  Hybrid Search  ->  Relevant Snippets
```

1. **First search** auto-ingests your working directory (extracts content, chunks, generates embeddings, stores in local Qdrant)
2. **Subsequent searches** are fast hybrid lookups (BM25 + semantic) — no re-ingestion needed
3. **Incremental updates** detect git changes and only re-embed modified files

## MCP Tools

8 primary tools + 6 deprecated aliases (old tool names still work with warnings):

| Tool | Purpose |
|------|---------|
| `search` | Semantic/hybrid search over indexed files |
| `search_by_file` | Search filtered by file path glob/substring |
| `ingest` | Manually trigger indexing (incremental by default, `force=True` for full) |
| `check_status` | Check if index is current or stale |
| `get_context` | Directory overview: languages, structure, dependencies, key files |
| `stats` | Index statistics (chunk count, collection, storage mode, provider) |
| `get_file_signatures` | Function/class signatures from files matching a pattern |
| `get_dependency_graph` | Internal import/dependency graph for the project |

Also supports `.env` file in the working directory.

## Embedding Providers

Zero-config by default. Choose your provider:

| Provider | Config | Notes |
|----------|--------|-------|
| **ONNX** (default) | None needed | Auto-downloads all-MiniLM-L6-v2 (23MB, 384-dim) |
| **Ollama** | `OMNI_RAG_EMBEDDING_PROVIDER=ollama` | Requires Ollama running with model pulled |
| **OpenAI** | `OMNI_RAG_EMBEDDING_PROVIDER=openai` + `OMNI_RAG_OPENAI_API_KEY=sk-...` | text-embedding-3-small |
| **Voyage** | `OMNI_RAG_EMBEDDING_PROVIDER=voyage` + `OMNI_RAG_VOYAGE_API_KEY=...` | voyage-code-3 (optimized for code) |

## Optional Extras

```bash
pip install omni-rag-mcp[pdf]    # PDF extraction (PyMuPDF)
pip install omni-rag-mcp[docx]   # Word document extraction
pip install omni-rag-mcp[image]  # Image/OCR extraction (Tesseract + Pillow)
pip install omni-rag-mcp[all]    # All optional extractors
```

## Storage

By default, uses Qdrant in local/on-disk mode — no Docker needed. All data is stored under `.omni-rag/` in the working directory:

- `.omni-rag/qdrant/` — Qdrant local storage
- `.omni-rag/models/` — ONNX embedding model cache

For remote Qdrant:

```bash
OMNI_RAG_QDRANT_MODE=remote
OMNI_RAG_QDRANT_HOST=your-host
OMNI_RAG_QDRANT_PORT=6333
```

## Configuration

All settings via environment variables with `OMNI_RAG_` prefix. See `config/.env.example` for the full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNI_RAG_EMBEDDING_PROVIDER` | `onnx` | Embedding provider: onnx, ollama, openai, voyage |
| `OMNI_RAG_QDRANT_MODE` | `local` | Storage mode: local or remote |
| `OMNI_RAG_INGESTION_TIMEOUT_HOURS` | `24` | Ingestion timeout in hours |
| `OMNI_RAG_WORKING_DIRECTORY` | cwd | Override working directory |
| `OMNI_RAG_HYBRID_SEARCH_ENABLED` | `true` | Enable BM25+semantic hybrid search |
| `OMNI_RAG_ENABLE_CODE_PLUGIN` | `true` | Enable code-specific tools |

Legacy `RAG_` prefix variables are still supported with deprecation warnings.

## Supported File Extensions

`.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.kt`, `.kts`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.rb`, `.php`, `.swift`, `.scala`, `.sh`, `.bash`, `.zsh`, `.fish`, `.html`, `.css`, `.scss`, `.less`, `.vue`, `.svelte`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, `.conf`, `.xml`, `.sql`, `.graphql`, `.proto`, `.md`, `.rst`, `.txt`, `.log`, `.adoc`, `.dockerfile`, `.env.example`, `.gitignore`, `.editorconfig`, `.gradle`, `.cmake`, `.makefile`

## Skipped Directories

`node_modules`, `__pycache__`, `venv`, `.venv`, `dist`, `build`, `.git`, `.codebase-rag`, `.rag-mcp`, `.omni-rag`, `.idea`, `.vscode`, `target`, `.gradle`

## Dependencies

Core dependencies:

- `mcp[cli]` ≥ 1.0.0
- `qdrant-client` ≥ 1.9.0
- `onnxruntime` ≥ 1.17.0
- `huggingface-hub` ≥ 0.20.0
- `tokenizers` ≥ 0.15.0
- `pydantic-settings` ≥ 2.0
- `pathspec` ≥ 0.12.0
- `requests` ≥ 2.31.0

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Health check
python scripts/health_check.py
```

## Manual MCP Registration

If `omni-rag-setup` doesn't work, add this to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "omni-rag": {
      "command": "omni-rag"
    }
  }
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
