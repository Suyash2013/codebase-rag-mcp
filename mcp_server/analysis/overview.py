"""Codebase overview generator — compressed project summary for Claude."""

import json
import logging
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from config.settings import settings

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

# Directories to skip — driven by settings.skip_directories


def _count_lines(filepath: Path, errors: list | None = None) -> int:
    """Count lines in a file (fast, no encoding errors)."""
    try:
        with open(filepath, "rb") as f:
            return sum(1 for _ in f)
    except Exception as e:
        log.warning("Failed to count lines in %s: %s", filepath, e)
        if errors is not None:
            errors.append(1)
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

        skip_dirs = set(settings.skip_directories)
        dirs = [
            e
            for e in entries
            if e.is_dir() and e.name not in skip_dirs and not e.name.startswith(".")
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


def _detect_dependencies(directory: str, errors: list | None = None) -> dict:
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
            except Exception as e:
                log.warning("Failed to parse pyproject.toml in %s: %s", directory, e)
                if errors is not None:
                    errors.append(1)

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
        except Exception as e:
            log.warning("Failed to parse package.json in %s: %s", directory, e)
            if errors is not None:
                errors.append(1)

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
        except Exception as e:
            log.warning("Failed to parse go.mod in %s: %s", directory, e)
            if errors is not None:
                errors.append(1)

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
        except Exception as e:
            log.warning("Failed to parse Cargo.toml in %s: %s", directory, e)
            if errors is not None:
                errors.append(1)

    return deps


def generate_overview(directory: str) -> dict:
    """Generate a compressed codebase overview.

    Returns a dict with: languages, structure, key_files, dependencies, stats.
    """
    directory = os.path.abspath(directory)
    base = Path(directory)

    errors: list = []

    # Language breakdown
    ext_counts: Counter[str] = Counter()
    ext_lines: Counter[str] = Counter()
    total_files = 0

    skip_dirs = set(settings.skip_directories)
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for f in files:
            fp = Path(root) / f
            ext = fp.suffix.lower() or f"({fp.name})"
            ext_counts[ext] += 1
            ext_lines[ext] += _count_lines(fp, errors)
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
    dependencies = _detect_dependencies(directory, errors)

    overview = {
        "directory": directory,
        "total_files": total_files,
        "languages": languages,
        "manifests": manifests,
        "key_files": dict(key_files),
        "dependencies": dependencies,
        "structure": tree[:100],  # Cap tree at 100 lines
    }

    error_count = len(errors)
    if error_count > 0:
        log.info("Overview generated with %d warnings", error_count)

    return overview


def _compute_fingerprint(directory: str) -> str:
    """Compute a lightweight fingerprint for cache invalidation.

    For git repos: returns the HEAD commit hash (fast, no directory walk).
    For non-git dirs: returns "file_count:total_size" based on top-level entries.
    """
    # Try git HEAD first — fast and accurate for version-controlled projects
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            if commit_hash:
                return commit_hash
    except Exception:
        pass

    # Fallback: count files and sum sizes of top-level entries only
    base = Path(directory)
    file_count = 0
    total_size = 0
    try:
        for entry in base.iterdir():
            if entry.is_file():
                file_count += 1
                try:
                    total_size += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return f"{file_count}:{total_size}"


def save_overview(directory: str, overview: dict) -> str:
    """Save overview and its fingerprint to .codebase-rag/overview.json."""
    cache_dir = Path(directory) / ".codebase-rag"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "overview.json"
    fingerprint = _compute_fingerprint(directory)
    payload = {"fingerprint": fingerprint, "overview": overview}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def load_cached_overview(directory: str) -> dict | None:
    """Load cached overview if it exists and the fingerprint matches.

    Returns None when the cache is missing or stale (fingerprint mismatch).
    """
    path = Path(directory) / ".codebase-rag" / "overview.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log.warning("Failed to load cached overview from %s: %s", path, e)
        return None

    # Support legacy cache format (no fingerprint key) — treat as stale
    if "fingerprint" not in payload or "overview" not in payload:
        log.debug("Cached overview at %s has no fingerprint; treating as stale", path)
        return None

    current_fingerprint = _compute_fingerprint(directory)
    if payload["fingerprint"] != current_fingerprint:
        log.debug(
            "Cached overview fingerprint mismatch for %s (cached=%s, current=%s); invalidating",
            directory,
            payload["fingerprint"],
            current_fingerprint,
        )
        return None

    return payload["overview"]
