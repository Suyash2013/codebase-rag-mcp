"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_codebase(tmp_path):
    """Create a temporary directory with sample code files for ingestion tests."""
    # Python file
    (tmp_path / "main.py").write_text(
        'def hello():\n    """Say hello."""\n    print("Hello, world!")\n\n\n'
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
    )

    # JavaScript file
    (tmp_path / "app.js").write_text(
        'const express = require("express");\n'
        "const app = express();\n\n"
        'app.get("/", (req, res) => {\n'
        '  res.send("Hello World");\n'
        "});\n"
    )

    # Config file
    (tmp_path / "config.yaml").write_text(
        "server:\n  host: localhost\n  port: 8080\n\n"
        "database:\n  url: postgres://localhost/mydb\n"
    )

    # Markdown docs
    (tmp_path / "README.md").write_text(
        "# Test Project\n\nA sample project for testing.\n\n## Usage\n\nRun `python main.py`.\n"
    )

    # .gitignore
    (tmp_path / ".gitignore").write_text(
        "__pycache__/\n*.pyc\nnode_modules/\n.env\n"
    )

    # Binary file (should be skipped)
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Nested directory
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "utils.py").write_text(
        'def format_name(first: str, last: str) -> str:\n    return f"{first} {last}"\n'
    )

    return tmp_path


@pytest.fixture
def mock_settings(tmp_path):
    """Provide test Settings with overridden values."""
    with patch.dict(os.environ, {
        "RAG_QDRANT_HOST": "localhost",
        "RAG_QDRANT_PORT": "6333",
        "RAG_QDRANT_COLLECTION": "test_codebase",
        "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
        "RAG_OLLAMA_EMBED_MODEL": "snowflake-arctic-embed:latest",
        "RAG_WORKING_DIRECTORY": str(tmp_path),
    }):
        from config.settings import Settings
        yield Settings()


@pytest.fixture
def sample_embedding():
    """A fake 1024-dimension embedding vector."""
    return [0.01 * i for i in range(1024)]
