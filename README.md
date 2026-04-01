# omni-rag-mcp

A general-purpose RAG MCP plugin for token-efficient semantic search over any directory of files. Auto-ingests the current working directory on first search and provides hybrid search (BM25 + semantic), directory overview, structural analysis, and dependency graphs.

Zero-config by default: local Qdrant storage, ONNX embeddings, no external services required. Supports code, markdown, PDFs, CSVs, and more via pluggable extractors.

## Quick Start

```bash
pip install omni-rag-mcp
omni-rag-setup
```

That's it. Restart Claude Code and the plugin auto-indexes your working directory on first search.

## How It Works

```
Your Files  ->  Extractors  ->  Chunking  ->  Embedding  ->  Qdrant (local)
                                                                 |
Claude Code ->  MCP Tool Call  ->  Hybrid Search  ->  Relevant Snippets
```

1. **First search** auto-ingests your working directory (extracts content, chunks, generates embeddings, stores in local Qdrant)
2. **Subsequent searches** are fast hybrid lookups (BM25 + semantic) -- no re-ingestion needed
3. **Incremental updates** detect git changes and only re-embed modified files

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search` | Hybrid search over indexed files (auto-ingests if needed) |
| `search_by_file` | Search filtered by file path pattern |
| `get_context` | Compressed directory overview (languages, structure, dependencies) |
| `get_file_signatures` | Function/class signatures without reading every file |
| `get_dependency_graph` | Internal import/dependency graph |
| `stats` | Index size and configuration |
| `ingest` | Manual re-index (incremental by default, `force=True` for full) |
| `check_status` | Is the index current? Any uncommitted changes? |

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

By default, uses Qdrant in local/on-disk mode -- no Docker needed. Data stored in `.omni-rag/` under your project directory.

For remote Qdrant:
```bash
OMNI_RAG_QDRANT_MODE=remote
OMNI_RAG_QDRANT_HOST=your-host
OMNI_RAG_QDRANT_PORT=6333
```

## Configuration

All settings via environment variables with `OMNI_RAG_` prefix. See `config/.env.example` for the full reference.

Legacy `RAG_` prefix variables are still supported with deprecation warnings.

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
