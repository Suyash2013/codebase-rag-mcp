# codebase-rag-mcp

A Claude Code MCP plugin that provides token-efficient semantic codebase search. Instead of Claude burning tokens with repeated Glob/Grep/Read cycles, it hits this plugin for semantic search first and gets relevant context cheaply.

## Quick Start

```bash
pip install codebase-rag-mcp
codebase-rag-setup
```

That's it. Restart Claude Code and the plugin auto-indexes your codebase on first search.

## How It Works

```
Your Code → Chunking → Embedding → Qdrant (local)
                                       ↓
Claude Code → MCP Tool Call → Semantic Search → Relevant Code Snippets
```

1. **First search** auto-ingests your working directory (chunks files, generates embeddings, stores in local Qdrant)
2. **Subsequent searches** are fast semantic lookups — no re-ingestion needed
3. **Incremental updates** detect git changes and only re-embed modified files

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_codebase` | Semantic search — "where is auth handled?", "how does retry work?" |
| `search_codebase_by_file` | Semantic search filtered by file path pattern |
| `get_codebase_context` | Compressed project overview (languages, structure, dependencies) |
| `get_file_signatures` | Function/class signatures without reading every file |
| `get_dependency_graph` | Internal import/dependency graph |
| `collection_stats` | Index size and configuration |
| `ingest_current_directory` | Manual re-index (incremental by default, `force=True` for full) |
| `check_index_status` | Is the index current? Any uncommitted changes? |

## Embedding Providers

Zero-config by default. Choose your provider:

| Provider | Config | Notes |
|----------|--------|-------|
| **ONNX** (default) | None needed | Auto-downloads all-MiniLM-L6-v2 (23MB, 384-dim) |
| **Ollama** | `RAG_EMBEDDING_PROVIDER=ollama` | Requires Ollama running with model pulled |
| **OpenAI** | `RAG_EMBEDDING_PROVIDER=openai` + `RAG_OPENAI_API_KEY=sk-...` | text-embedding-3-small |
| **Voyage** | `RAG_EMBEDDING_PROVIDER=voyage` + `RAG_VOYAGE_API_KEY=...` | voyage-code-3 (optimized for code) |

## Storage

By default, uses Qdrant in local/on-disk mode — no Docker needed. Data stored in `.codebase-rag/` under your project directory.

For remote Qdrant:
```bash
RAG_QDRANT_MODE=remote
RAG_QDRANT_HOST=your-host
RAG_QDRANT_PORT=6333
```

## Configuration

All settings via environment variables with `RAG_` prefix. See `config/.env.example` for the full reference.

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

If `codebase-rag-setup` doesn't work, add this to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "codebase-rag"
    }
  }
}
```
