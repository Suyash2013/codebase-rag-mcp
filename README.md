# codebase-rag-mcp

Claude Code MCP plugin for semantic codebase search via Qdrant + Ollama.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│ INGESTION (auto or manual)                           │
│                                                      │
│ Working Dir ──► Chunk ──► Embed ──► Qdrant           │
│   (.gitignore    (1000/200)  (snowflake-    (cosine  │
│    aware)                     arctic-embed)  search) │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│ RETRIEVAL (MCP server — Claude Code plugin)          │
│                                                      │
│ Claude Code ──► MCP tool ──► this server             │
│                                  │                   │
│                  ┌───────────────┘                   │
│                  ▼                                   │
│           Ollama (embed query)                       │
│                  │                                   │
│                  ▼                                   │
│           Qdrant (similarity search)                 │
│                  │                                   │
│                  ▼                                   │
│           code chunks ──► Claude Code                │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ LANGFLOW (optional bidirectional)                    │
│                                                      │
│ Langflow ◄──► MCP Bridge Component ◄──► Qdrant      │
│ Langflow ◄──── trigger_langflow_ingestion (MCP tool) │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Qdrant** running locally:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Ollama** running with snowflake-arctic-embed:
   ```bash
   ollama pull snowflake-arctic-embed
   ```

3. **Python 3.10+** with `uv` (recommended) or `pip`.

## Setup

```bash
# Install dependencies
uv sync

# Verify services
python scripts/health_check.py

# Test the server
uv run mcp_server/server.py
```

## Register with Claude Code

### Quick setup
```powershell
claude mcp add codebase-rag -s user -- uv run --directory "E:\path\to\codebase-rag-mcp" python mcp_server/server.py
```

### Manual setup
Add to `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "uv",
      "args": ["run", "--directory", "<path-to-this-repo>", "python", "mcp_server/server.py"],
      "env": {
        "RAG_QDRANT_HOST": "localhost",
        "RAG_QDRANT_PORT": "6333",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest"
      }
    }
  }
}
```

### Verify
In Claude Code, run `/mcp` — you should see `codebase-rag` with 6 tools.

## Available Tools

| Tool | Description |
|------|-------------|
| `search_codebase(query, n_results)` | Semantic search. Auto-ingests on first use. Preferred over file search when no local changes. |
| `search_codebase_by_file(query, file_pattern, n_results)` | Search filtered by file path substring. |
| `collection_stats()` | Qdrant collection statistics. |
| `ingest_current_directory(force)` | Manual ingestion. Use `force=True` to re-index. |
| `check_index_status()` | Check if indexed + detect local git changes. |
| `trigger_langflow_ingestion(directory)` | Trigger Langflow pipeline via REST API. |

## Configuration

All via env vars with `RAG_` prefix. See `config/.env.example` for full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_QDRANT_HOST` | `localhost` | Qdrant server host |
| `RAG_QDRANT_PORT` | `6333` | Qdrant server port |
| `RAG_QDRANT_COLLECTION` | `codebase` | Collection name |
| `RAG_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `RAG_OLLAMA_EMBED_MODEL` | `snowflake-arctic-embed:latest` | Embedding model |
| `RAG_INGESTION_TIMEOUT_HOURS` | `24` | Max ingestion time |
| `RAG_CHUNK_SIZE` | `1000` | Text chunk size |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_WORKING_DIRECTORY` | (cwd) | Override working directory |

## Project Structure

```
├── mcp_server/           MCP server package
│   ├── server.py         Entrypoint (FastMCP)
│   ├── embeddings.py     Ollama embedding helper
│   ├── qdrant_client.py  Qdrant operations
│   ├── ingestion.py      Auto-ingestion engine
│   └── tools/            MCP tool definitions
├── flows/                Langflow JSON exports
├── components/           Custom Langflow components
├── config/               Settings (pydantic-settings)
├── scripts/              Health check, validation
└── tests/                pytest suite
```

## Key Behaviors

- **Auto-ingestion**: First search in an un-indexed directory triggers automatic ingestion
- **Search priority**: Preferred over file search when codebase is indexed and no uncommitted changes
- **.gitignore aware**: Ingestion skips files matching .gitignore patterns
- **24-hour timeout**: Long-running ingestion has a configurable timeout (default 24h)
- **Incremental**: Re-ingestion replaces previous index for the directory
