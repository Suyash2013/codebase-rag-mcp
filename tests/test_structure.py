"""Tests for structural analysis (signatures and dependency graph)."""

import pytest

from mcp_server.analysis.structure import (
    build_dependency_graph,
    extract_imports,
    extract_signatures,
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
    f.write_text("""
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
""")
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


# ---------------------------------------------------------------------------
# Go signature extraction
# ---------------------------------------------------------------------------


@pytest.fixture
def go_file(tmp_path):
    """Create a sample Go file."""
    f = tmp_path / "main.go"
    f.write_text("""
package main

import "fmt"

type Server struct {
    Host string
    Port int
}

type Handler interface {
    ServeHTTP(w ResponseWriter, r *Request)
}

func NewServer(host string, port int) *Server {
    return &Server{Host: host, Port: port}
}

func (s *Server) Start() error {
    fmt.Println("Starting server")
    return nil
}

func (s Server) GetHost() string {
    return s.Host
}
""")
    return str(f)


def test_extract_go_functions(go_file):
    """Should extract Go functions without receivers."""
    sigs = extract_signatures(go_file)
    names = [s["name"] for s in sigs]
    assert "NewServer" in names
    new_server = next(s for s in sigs if s["name"] == "NewServer")
    assert new_server["type"] == "function"
    assert len(new_server["params"]) > 0


def test_extract_go_methods_with_receivers(go_file):
    """Should extract Go methods with receiver type in the name."""
    sigs = extract_signatures(go_file)
    names = [s["name"] for s in sigs]
    assert "Server.Start" in names
    assert "Server.GetHost" in names
    start = next(s for s in sigs if s["name"] == "Server.Start")
    assert start["type"] == "method"


def test_extract_go_structs(go_file):
    """Should extract Go struct declarations."""
    sigs = extract_signatures(go_file)
    struct_sigs = [s for s in sigs if s["type"] == "struct"]
    struct_names = [s["name"] for s in struct_sigs]
    assert "Server" in struct_names


def test_extract_go_interfaces(go_file):
    """Should extract Go interface declarations."""
    sigs = extract_signatures(go_file)
    iface_sigs = [s for s in sigs if s["type"] == "interface"]
    iface_names = [s["name"] for s in iface_sigs]
    assert "Handler" in iface_names


# ---------------------------------------------------------------------------
# Java / Kotlin signature extraction
# ---------------------------------------------------------------------------


@pytest.fixture
def java_file(tmp_path):
    """Create a sample Java file."""
    f = tmp_path / "App.java"
    f.write_text("""
public class App extends BaseApp {

    public static void main(String[] args) {
        System.out.println("Hello");
    }

    private String getName(int id) {
        return "name";
    }
}
""")
    return str(f)


@pytest.fixture
def kotlin_file(tmp_path):
    """Create a sample Kotlin file."""
    f = tmp_path / "Main.kt"
    f.write_text("""
data class User(val name: String, val age: Int)

open class Repository : BaseRepository {
    fun findById(id: Int): User {
        return User("test", 0)
    }

    suspend fun fetchAll(): List {
        return emptyList()
    }
}

fun greet(name: String): String {
    return "Hello, $name"
}
""")
    return str(f)


def test_extract_java_classes(java_file):
    """Should extract Java class declarations with base class."""
    sigs = extract_signatures(java_file)
    names = [s["name"] for s in sigs]
    assert "App" in names
    app_cls = next(s for s in sigs if s["name"] == "App")
    assert app_cls["type"] == "class"
    assert "BaseApp" in app_cls.get("bases", [])


def test_extract_java_methods(java_file):
    """Should extract Java methods with return types and params."""
    sigs = extract_signatures(java_file)
    names = [s["name"] for s in sigs]
    assert "main" in names
    assert "getName" in names
    get_name = next(s for s in sigs if s["name"] == "getName")
    assert get_name["type"] == "method"
    assert get_name["return_type"] == "String"


def test_extract_kotlin_classes(kotlin_file):
    """Should extract Kotlin class declarations including data classes."""
    sigs = extract_signatures(kotlin_file)
    names = [s["name"] for s in sigs]
    assert "User" in names
    assert "Repository" in names


def test_extract_kotlin_functions(kotlin_file):
    """Should extract Kotlin functions including suspend functions."""
    sigs = extract_signatures(kotlin_file)
    names = [s["name"] for s in sigs]
    assert "greet" in names
    assert "findById" in names
    assert "fetchAll" in names
    greet = next(s for s in sigs if s["name"] == "greet")
    assert greet["type"] == "function"
    assert greet.get("return_type") == "String"
