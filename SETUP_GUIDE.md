# Setup Guide

## Prerequisites

1. **Qdrant** — Vector database
   ```bash
   docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
   ```

2. **Ollama** — Local embeddings
   ```bash
   ollama pull snowflake-arctic-embed
   ```

3. **Python 3.10+** with `uv`
   ```bash
   pip install uv
   ```

## MCP Server Setup

```bash
cd codebase-rag-mcp

# Install dependencies
uv sync

# Verify prerequisites
python scripts/health_check.py

# Register with Claude Code
claude mcp add codebase-rag -s user -- uv run --directory "$(pwd)" python mcp_server/server.py
```

## Langflow Custom Components

### Import Components
In Langflow UI:
1. Click "New Custom Component" (bottom left)
2. Copy-paste code from each file in `components/`:
   - `file_reader_component.py` — Read files from project
   - `file_writer_component.py` — Write files with backup/dry-run
   - `model_router_component.py` — Smart API/local model routing
   - `directory_component.py` — Directory traversal
   - `mcp_bridge_component.py` — Semantic search via Qdrant (MCP bridge)
3. Save each component

### MCP Bridge Component
The `mcp_bridge_component.py` gives Langflow flows access to the same Qdrant index used by Claude Code:
- Drag "Codebase Search (Qdrant)" onto your flow canvas
- Connect its output to your agent or prompt
- Configure Qdrant host/port and collection name in advanced settings
- Uses the same snowflake-arctic-embed model as the MCP server

### Configure Flow
1. Drag components to canvas
2. Enable "Tool Mode" on components you want agents to use
3. Connect tool outputs to Agent's tools input

## Configuration

Copy `config/.env.example` to `.env` and adjust values:
```bash
cp config/.env.example .env
```

All settings use the `RAG_` prefix. See `config/.env.example` for the full list.

## Verify

```bash
# Check services
python scripts/health_check.py

# Validate flow files
python scripts/validate_flows.py

# Run tests
pytest tests/ -v

# In Claude Code, verify the plugin
# /mcp → should show codebase-rag with 6 tools
```
