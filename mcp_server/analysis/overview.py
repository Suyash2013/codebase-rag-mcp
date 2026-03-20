"""Codebase overview generator — compressed project summary for Claude."""

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger("codebase-rag-mcp")

# Known project manifest files and what they indicate
_MANIFEST_FILES = {
    "pyproject.toml": "Python (modern)",
    "setup.py": "Python (legacy)",
    "requirements.txt": "Python",
    "Pipfile": "Python (pipenv)",
    "package.json": "JavaScript/TypeScript (npm)",
    "yarn.lock": "JavaScript/TypeScript (yarn)",
    "pnpm-lock.yaml": "JavaScript/TypeScript (pnpm)",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "build.gradle": "Java/Kotlin (Gradle)",
    "build.gradle.kts": "Kotlin (Gradle KTS)",
    "pom.xml": "Java (Maven)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "Package.swift": "Swift",
    "CMakeLists.txt": "C/C++ (CMake)",
    "Makefile": "C/C++ or general",
}

# Key file patterns that indicate project structure
_KEY_FILE_PATTERNS = {
    "entry_points": [
        "main.py",
        "app.py",
        "server.py",
        "index.js",
        "index.ts",
        "main.go",
        "main.rs",
        "Main.java",
        "Main.kt",
    ],
    "config": [
        "pyproject.toml",
        "package.json",
        "tsconfig.json",
        "go.mod",
        "Cargo.toml",
        ".env.example",
        "docker-compose.yml",
        "Dockerfile",
    ],
    "tests": ["conftest.py", "jest.config.js", "pytest.ini", "test_*.py"],
    "docs": ["README.md", "CHANGELOG.md", "CONTRIBUTING.md", "docs/"],
    "ci_cd": [".github/workflows/", ".gitlab-ci.yml", "Jenkinsfile"],
}

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


def _count_lines(filepath: Path) -> int:
    """Count lines in a file (fast, no encoding errors)."""
    try:
        with open(filepath, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _build_dir_tree(directory: str, max_depth: int = 3) -> list[str]:
    """Build a depth-limited directory tree listing."""
    tree = []
    base = Path(directory)

    def _walk(path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return

        dirs = [
            e
            for e in entries
            if e.is_dir() and e.name not in SKIP_DIRS and not e.name.startswith(".")
        ]
        files = [e for e in entries if e.is_file()]

        for d in dirs:
            tree.append(f"{prefix}{d.name}/")
            _walk(d, depth + 1, prefix + "  ")

        # Only show files at top 2 levels to keep it compact
        if depth <= 2:
            for f in files[:20]:  # Cap at 20 files per directory
                tree.append(f"{prefix}{f.name}")
            if len(files) > 20:
                tree.append(f"{prefix}... ({len(files) - 20} more files)")

    _walk(base, 0)
    return tree


def _detect_dependencies(directory: str) -> dict:
    """Parse manifest files to extract dependency information."""
    deps = {}

    # pyproject.toml
    pyproject = Path(directory) / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                tomllib = None

        if tomllib:
            try:
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                project = data.get("project", {})
                deps["python"] = {
                    "name": project.get("name", "unknown"),
                    "version": project.get("version", "unknown"),
                    "dependencies": project.get("dependencies", []),
                    "python_requires": project.get("requires-python", ""),
                }
            except Exception:
                pass

    # package.json
    pkg_json = Path(directory) / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, encoding="utf-8") as f:
                data = json.load(f)
            deps["node"] = {
                "name": data.get("name", "unknown"),
                "version": data.get("version", "unknown"),
                "dependencies": list(data.get("dependencies", {}).keys()),
                "devDependencies": list(data.get("devDependencies", {}).keys()),
            }
        except Exception:
            pass

    # go.mod
    go_mod = Path(directory) / "go.mod"
    if go_mod.exists():
        try:
            text = go_mod.read_text(encoding="utf-8")
            lines = text.splitlines()
            module = next(
                (line.split()[1] for line in lines if line.startswith("module ")), "unknown"
            )
            requires = [
                line.strip().split()[0]
                for line in lines
                if line.strip() and not line.startswith(("module", "go ", ")", "require"))
            ]
            deps["go"] = {"module": module, "requires": requires[:20]}
        except Exception:
            pass

    # Cargo.toml
    cargo = Path(directory) / "Cargo.toml"
    if cargo.exists():
        try:
            if tomllib:
                with open(cargo, "rb") as f:
                    data = tomllib.load(f)
                package = data.get("package", {})
                deps["rust"] = {
                    "name": package.get("name", "unknown"),
                    "version": package.get("version", "unknown"),
                    "dependencies": list(data.get("dependencies", {}).keys()),
                }
        except Exception:
            pass

    return deps


def generate_overview(directory: str) -> dict:
    """Generate a compressed codebase overview.

    Returns a dict with: languages, structure, key_files, dependencies, stats.
    """
    directory = os.path.abspath(directory)
    base = Path(directory)

    # Language breakdown
    ext_counts: Counter[str] = Counter()
    ext_lines: Counter[str] = Counter()
    total_files = 0

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for f in files:
            fp = Path(root) / f
            ext = fp.suffix.lower() or f"({fp.name})"
            ext_counts[ext] += 1
            ext_lines[ext] += _count_lines(fp)
            total_files += 1

    # Top languages by file count
    languages = [
        {"extension": ext, "files": count, "lines": ext_lines[ext]}
        for ext, count in ext_counts.most_common(15)
    ]

    # Directory tree
    tree = _build_dir_tree(directory)

    # Key files detection
    key_files = defaultdict(list)
    for category, patterns in _KEY_FILE_PATTERNS.items():
        for pattern in patterns:
            if pattern.endswith("/"):
                # Directory pattern
                if (base / pattern.rstrip("/")).is_dir():
                    key_files[category].append(pattern)
            else:
                # Check both exact match and glob
                matches = list(base.glob(f"**/{pattern}"))
                for m in matches[:5]:
                    key_files[category].append(str(m.relative_to(base)))

    # Manifest detection
    manifests = []
    for filename, lang in _MANIFEST_FILES.items():
        if (base / filename).exists():
            manifests.append({"file": filename, "language": lang})

    # Dependencies
    dependencies = _detect_dependencies(directory)

    overview = {
        "directory": directory,
        "total_files": total_files,
        "languages": languages,
        "manifests": manifests,
        "key_files": dict(key_files),
        "dependencies": dependencies,
        "structure": tree[:100],  # Cap tree at 100 lines
    }

    return overview


def save_overview(directory: str, overview: dict) -> str:
    """Save overview to .codebase-rag/overview.json."""
    cache_dir = Path(directory) / ".codebase-rag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "overview.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    return str(path)


def load_cached_overview(directory: str) -> dict | None:
    """Load cached overview if it exists."""
    path = Path(directory) / ".codebase-rag" / "overview.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
