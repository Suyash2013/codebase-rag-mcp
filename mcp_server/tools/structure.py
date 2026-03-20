"""MCP tools for structural codebase analysis."""

import os
from pathlib import Path

from config.settings import settings
from mcp_server.analysis.structure import (
    build_dependency_graph,
    extract_signatures,
)

# Directories to skip
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


def get_file_signatures(file_pattern: str = "") -> str:
    """Get function and class signatures from files matching a pattern.

    Use this to understand the API surface of a module without reading
    every file. Returns function names, parameters, return types, and
    class hierarchies.

    More structured than semantic search — gives you the exact interface
    of code modules.

    Args:
        file_pattern: Substring to match in file paths (case-insensitive).
                      E.g. "models", "api/routes", ".py", "utils".
                      Empty string matches all files.
    """
    directory = settings.get_working_directory()

    try:
        base = Path(directory)
        results = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for f in files:
                fp = Path(root) / f
                rel_path = str(fp.relative_to(base))

                if file_pattern and file_pattern.lower() not in rel_path.lower():
                    continue

                sigs = extract_signatures(str(fp))
                if sigs:
                    results.append({"file": rel_path, "signatures": sigs})

        if not results:
            return f"No signatures found for pattern '{file_pattern}'"

        return _format_signatures(results)

    except Exception as exc:
        return f"Error extracting signatures: {exc}"


def get_dependency_graph(file_pattern: str = "") -> str:
    """Get the import/dependency graph for the codebase.

    Shows which files import from which other files within the project.
    Use this to understand module boundaries and find the right place
    to make changes.

    Args:
        file_pattern: Optional substring to filter files (case-insensitive).
                      Empty string analyzes the entire project.
    """
    directory = settings.get_working_directory()

    try:
        graph = build_dependency_graph(directory, file_pattern or None)

        if not graph:
            return f"No internal dependencies found{' for pattern: ' + file_pattern if file_pattern else ''}"

        return _format_graph(graph)

    except Exception as exc:
        return f"Error building dependency graph: {exc}"


def _format_signatures(results: list[dict]) -> str:
    """Format signature results into readable text."""
    parts = [f"Found signatures in {len(results)} files:\n"]

    for file_result in results[:30]:  # Cap output
        parts.append(f"\n## {file_result['file']}")
        for sig in file_result["signatures"]:
            sig_type = sig.get("type", "unknown")
            name = sig["name"]
            line = sig.get("line", "?")

            if sig_type in ("function", "async_function", "method", "arrow_function"):
                params = ", ".join(sig.get("params", []))
                ret = f" -> {sig['return_type']}" if sig.get("return_type") else ""
                prefix = "async " if sig_type == "async_function" else ""
                parts.append(f"  L{line}: {prefix}def {name}({params}){ret}")
            elif sig_type == "class":
                bases = f"({', '.join(sig.get('bases', []))})" if sig.get("bases") else ""
                parts.append(f"  L{line}: class {name}{bases}")
            elif sig_type in ("interface", "struct"):
                bases = f" extends {', '.join(sig.get('bases', []))}" if sig.get("bases") else ""
                parts.append(f"  L{line}: {sig_type} {name}{bases}")

            if sig.get("docstring"):
                parts.append(f"          # {sig['docstring']}")

    if len(results) > 30:
        parts.append(f"\n... and {len(results) - 30} more files")

    return "\n".join(parts)


def _format_graph(graph: dict) -> str:
    """Format dependency graph into readable text."""
    parts = [f"Internal dependency graph ({len(graph)} files with dependencies):\n"]

    for file_path, deps in sorted(graph.items()):
        parts.append(f"  {file_path}")
        for dep in deps:
            parts.append(f"    → {dep}")

    return "\n".join(parts)
