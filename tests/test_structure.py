"""Tests for structural analysis (signatures and dependency graph)."""

from pathlib import Path

import pytest

from mcp_server.analysis.structure import (
    extract_signatures,
    extract_imports,
    build_dependency_graph,
)


@pytest.fixture
def python_file(tmp_path):
    """Create a sample Python file."""
    f = tmp_path / "sample.py"
    f.write_text('''
import os
from pathlib import Path

class MyClass(BaseClass):
    """A sample class."""

    def method(self, x: int) -> str:
        """Do something."""
        return str(x)

async def async_handler(request: dict) -> dict:
    """Handle a request."""
    return {}

def simple_func(a, b):
    return a + b
''')
    return str(f)


@pytest.fixture
def js_file(tmp_path):
    """Create a sample JS file."""
    f = tmp_path / "app.js"
    f.write_text('''
import { Router } from 'express';
const axios = require('axios');

export class UserService extends BaseService {
    async findUser(id) {
        return null;
    }
}

export function handleRequest(req, res) {
    res.send('ok');
}

const processData = async (data, options) => {
    return data;
};
''')
    return str(f)


def test_extract_python_signatures(python_file):
    """Should extract functions and classes from Python."""
    sigs = extract_signatures(python_file)
    names = [s["name"] for s in sigs]

    assert "MyClass" in names
    assert "method" in names
    assert "async_handler" in names
    assert "simple_func" in names

    # Check class details
    cls = next(s for s in sigs if s["name"] == "MyClass")
    assert cls["type"] == "class"
    assert "BaseClass" in cls.get("bases", [])

    # Check async function
    async_fn = next(s for s in sigs if s["name"] == "async_handler")
    assert async_fn["type"] == "async_function"
    assert "request: dict" in async_fn["params"]


def test_extract_python_imports(python_file):
    """Should extract Python imports."""
    imports = extract_imports(python_file)
    assert "os" in imports
    assert "pathlib" in imports


def test_extract_js_signatures(js_file):
    """Should extract functions and classes from JavaScript."""
    sigs = extract_signatures(js_file)
    names = [s["name"] for s in sigs]

    assert "UserService" in names
    assert "handleRequest" in names
    assert "processData" in names


def test_extract_js_imports(js_file):
    """Should extract JS imports."""
    imports = extract_imports(js_file)
    assert "express" in imports
    assert "axios" in imports


def test_extract_signatures_nonexistent():
    """Should return empty for non-existent files."""
    sigs = extract_signatures("/nonexistent/file.py")
    assert sigs == []


def test_build_dependency_graph(tmp_path):
    """Should build internal dependency graph."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "__init__.py").write_text("")
    (tmp_path / "config" / "settings.py").write_text("VALUE = 1\n")
    (tmp_path / "main.py").write_text("from config.settings import VALUE\n")

    graph = build_dependency_graph(str(tmp_path))
    assert "main.py" in graph
    assert any("config" in dep for dep in graph["main.py"])


def test_build_dependency_graph_empty(tmp_path):
    """Should return empty dict for project with no internal deps."""
    (tmp_path / "standalone.py").write_text("import os\nimport sys\n")

    graph = build_dependency_graph(str(tmp_path))
    assert "standalone.py" not in graph  # Only external imports
