# Contributing to codebase-rag-mcp

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone <repo-url>
cd codebase-rag-mcp
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Running Checks

```bash
# Lint
ruff check .

# Format check
ruff format --check .

# Type check
mypy mcp_server config

# Tests with coverage
pytest --cov
```

## PR Guidelines

- All CI checks must pass before merge (lint, typecheck, tests, security audit).
- Claude Code will automatically review PRs and leave comments.
- Keep PRs focused — one logical change per PR.

## Adding a New Embedding Provider

1. Create `mcp_server/embeddings/your_provider.py` implementing the `EmbeddingProvider` base class.
2. Implement the `embed()` and `get_dimension()` methods.
3. Add any new settings to `config/settings.py` with the `RAG_` prefix.
4. Register the provider in `mcp_server/embeddings/factory.py`.
5. Add tests in `tests/test_embeddings.py`.
6. Document the new `RAG_*` variables in `config/.env.example`.
