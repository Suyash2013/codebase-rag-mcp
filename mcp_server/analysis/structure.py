"""Lightweight structural analysis — AST for Python, regex for others."""

import ast
import logging
import os
import re
from pathlib import Path

log = logging.getLogger("codebase-rag-mcp")

# Directories to skip during analysis
SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "dist",
    "build",
    ".git",
    ".codebase-rag",
    ".idea",
    ".vscode",
    "target",
    ".gradle",
}


def extract_signatures(file_path: str) -> list[dict]:
    """Extract function and class signatures from a file.

    Uses Python AST for .py files, regex patterns for others.
    Returns list of dicts with: name, type, line, params, return_type, docstring.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if path.suffix == ".py":
        return _extract_python_signatures(text)
    elif path.suffix in {".js", ".ts", ".tsx", ".jsx"}:
        return _extract_js_ts_signatures(text)
    elif path.suffix in {".go"}:
        return _extract_go_signatures(text)
    elif path.suffix in {".java", ".kt", ".kts"}:
        return _extract_jvm_signatures(text)
    else:
        return _extract_generic_signatures(text)


def _extract_python_signatures(source: str) -> list[dict]:
    """Extract signatures from Python source using AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in node.args.args:
                param = arg.arg
                if arg.annotation:
                    param += f": {ast.unparse(arg.annotation)}"
                params.append(param)

            return_type = ast.unparse(node.returns) if node.returns else None
            docstring = ast.get_docstring(node)

            signatures.append(
                {
                    "name": node.name,
                    "type": "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function",
                    "line": node.lineno,
                    "params": params,
                    "return_type": return_type,
                    "docstring": docstring[:100] if docstring else None,
                }
            )

        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            docstring = ast.get_docstring(node)

            signatures.append(
                {
                    "name": node.name,
                    "type": "class",
                    "line": node.lineno,
                    "bases": bases,
                    "docstring": docstring[:100] if docstring else None,
                }
            )

    return signatures


def _extract_js_ts_signatures(source: str) -> list[dict]:
    """Extract signatures from JS/TS using regex."""
    signatures = []

    # Functions: function name(params) or const name = (params) =>
    for m in re.finditer(
        r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        signatures.append(
            {
                "name": m.group(1),
                "type": "function",
                "line": line,
                "params": [p.strip() for p in m.group(2).split(",") if p.strip()],
            }
        )

    # Arrow functions: const name = (params) =>
    for m in re.finditer(
        r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*\w+)?\s*=>",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        signatures.append(
            {
                "name": m.group(1),
                "type": "arrow_function",
                "line": line,
                "params": [p.strip() for p in m.group(2).split(",") if p.strip()],
            }
        )

    # Classes
    for m in re.finditer(
        r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        sig = {"name": m.group(1), "type": "class", "line": line}
        if m.group(2):
            sig["bases"] = [m.group(2)]
        signatures.append(sig)

    # Interfaces (TypeScript)
    for m in re.finditer(
        r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        sig = {"name": m.group(1), "type": "interface", "line": line}
        if m.group(2):
            sig["bases"] = [b.strip() for b in m.group(2).split(",")]
        signatures.append(sig)

    return signatures


def _extract_go_signatures(source: str) -> list[dict]:
    """Extract signatures from Go using regex."""
    signatures = []

    for m in re.finditer(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s+([^{]+))?",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        name = m.group(3)
        receiver = m.group(2)
        params = [p.strip() for p in m.group(4).split(",") if p.strip()]
        return_type = m.group(5).strip() if m.group(5) else None

        sig = {
            "name": f"{receiver}.{name}" if receiver else name,
            "type": "method" if receiver else "function",
            "line": line,
            "params": params,
        }
        if return_type:
            sig["return_type"] = return_type
        signatures.append(sig)

    for m in re.finditer(r"type\s+(\w+)\s+struct", source):
        line = source[: m.start()].count("\n") + 1
        signatures.append({"name": m.group(1), "type": "struct", "line": line})

    for m in re.finditer(r"type\s+(\w+)\s+interface", source):
        line = source[: m.start()].count("\n") + 1
        signatures.append({"name": m.group(1), "type": "interface", "line": line})

    return signatures


def _extract_jvm_signatures(source: str) -> list[dict]:
    """Extract signatures from Java/Kotlin using regex."""
    signatures = []

    # Java/Kotlin classes
    for m in re.finditer(
        r"(?:public|private|protected|internal)?\s*(?:abstract|open|data)?\s*class\s+(\w+)(?:\s*(?:extends|:)\s*(\w+))?",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        sig = {"name": m.group(1), "type": "class", "line": line}
        if m.group(2):
            sig["bases"] = [m.group(2)]
        signatures.append(sig)

    # Java methods
    for m in re.finditer(
        r"(?:public|private|protected)\s+(?:static\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        signatures.append(
            {
                "name": m.group(2),
                "type": "method",
                "line": line,
                "return_type": m.group(1),
                "params": [p.strip() for p in m.group(3).split(",") if p.strip()],
            }
        )

    # Kotlin functions
    for m in re.finditer(
        r"(?:fun|suspend\s+fun)\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+))?",
        source,
    ):
        line = source[: m.start()].count("\n") + 1
        sig = {
            "name": m.group(1),
            "type": "function",
            "line": line,
            "params": [p.strip() for p in m.group(2).split(",") if p.strip()],
        }
        if m.group(3):
            sig["return_type"] = m.group(3)
        signatures.append(sig)

    return signatures


def _extract_generic_signatures(source: str) -> list[dict]:
    """Best-effort signature extraction using common patterns."""
    signatures = []

    # Generic function-like patterns
    for m in re.finditer(r"(?:def|fn|func|function)\s+(\w+)\s*\(([^)]*)\)", source):
        line = source[: m.start()].count("\n") + 1
        signatures.append(
            {
                "name": m.group(1),
                "type": "function",
                "line": line,
                "params": [p.strip() for p in m.group(2).split(",") if p.strip()],
            }
        )

    return signatures


def extract_imports(file_path: str) -> list[str]:
    """Extract import statements from a file.

    Returns list of imported module/package names.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if path.suffix == ".py":
        return _extract_python_imports(text)
    elif path.suffix in {".js", ".ts", ".tsx", ".jsx"}:
        return _extract_js_imports(text)
    elif path.suffix == ".go":
        return _extract_go_imports(text)
    else:
        return []


def _extract_python_imports(source: str) -> list[str]:
    """Extract Python imports using AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return imports


def _extract_js_imports(source: str) -> list[str]:
    """Extract JS/TS imports using regex."""
    imports = []

    # ES6 imports
    for m in re.finditer(r"""import\s+.*?from\s+['"]([^'"]+)['"]""", source):
        imports.append(m.group(1))

    # require()
    for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""", source):
        imports.append(m.group(1))

    return imports


def _extract_go_imports(source: str) -> list[str]:
    """Extract Go imports using regex."""
    imports = []
    for m in re.finditer(r'"([^"]+)"', source):
        path = m.group(1)
        if "/" in path or path in {"fmt", "os", "io", "log", "net", "sync", "time", "context"}:
            imports.append(path)
    return imports


def build_dependency_graph(
    directory: str,
    file_pattern: str | None = None,
) -> dict[str, list[str]]:
    """Build a file-level dependency graph.

    Returns dict mapping each file (relative path) to the files it imports from.
    Only includes internal imports (within the project).
    """
    base = Path(directory)
    graph = {}

    # Collect all project files
    project_modules = set()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            fp = Path(root) / f
            rel = str(fp.relative_to(base))
            if file_pattern and file_pattern.lower() not in rel.lower():
                continue
            project_modules.add(rel)

    # Normalize project modules to forward slashes for matching
    normalized_modules = {m.replace("\\", "/"): m for m in project_modules}

    # Build graph
    for rel_path in project_modules:
        fp = base / rel_path
        imports = extract_imports(str(fp))

        # Resolve imports to project files
        internal_deps = []
        for imp in imports:
            # Convert import to possible file paths (always uses forward slashes)
            candidates = _import_to_paths(imp, fp.suffix)
            for candidate in candidates:
                if candidate in normalized_modules:
                    original = normalized_modules[candidate]
                    if original != rel_path:
                        internal_deps.append(original)
                        break

        if internal_deps:
            graph[rel_path] = internal_deps

    return graph


def _import_to_paths(import_name: str, source_suffix: str) -> list[str]:
    """Convert an import name to possible file paths."""
    candidates = []

    if source_suffix == ".py":
        # Python: "config.settings" -> "config/settings.py"
        path = import_name.replace(".", "/")
        candidates.append(f"{path}.py")
        candidates.append(f"{path}/__init__.py")

    elif source_suffix in {".js", ".ts", ".tsx", ".jsx"}:
        # JS/TS: relative imports only
        if import_name.startswith("."):
            clean = import_name.lstrip("./")
            for ext in [".ts", ".tsx", ".js", ".jsx"]:
                candidates.append(f"{clean}{ext}")
            candidates.append(f"{clean}/index.ts")
            candidates.append(f"{clean}/index.js")

    elif source_suffix == ".go":
        # Go: package-based, harder to resolve
        pass

    return candidates
