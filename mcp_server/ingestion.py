"""Auto-ingestion engine — walks a directory, chunks files, embeds, stores in Qdrant."""

import hashlib
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pathspec

from config.settings import settings
from mcp_server.embeddings import get_embedding, get_embedding_dimension
from mcp_server.qdrant_client import (
    delete_directory_points,
    delete_file_points,
    ensure_collection,
    is_directory_indexed,
    upsert_chunks,
)

log = logging.getLogger("codebase-rag-mcp")

# File extensions considered as text (code, config, docs)
TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".vue",
    ".svelte",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".sql",
    ".graphql",
    ".proto",
    ".md",
    ".rst",
    ".txt",
    ".adoc",
    ".dockerfile",
    ".env.example",
    ".gitignore",
    ".editorconfig",
    ".gradle",
    ".cmake",
    ".makefile",
}

# Files without extensions that are typically text
TEXT_FILENAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "Jenkinsfile",
    "Procfile",
    "Vagrantfile",
    "Gemfile",
    "Rakefile",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
}

# Directories to always skip
SKIP_DIRS = {
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "dist",
    "build",
    ".git",
    ".codebase-rag",
}


def _load_gitignore(directory: str) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the directory."""
    gitignore_path = Path(directory) / ".gitignore"
    if not gitignore_path.exists():
        return None
    with open(gitignore_path, encoding="utf-8", errors="ignore") as f:
        return pathspec.PathSpec.from_lines("gitignore", f)


def _is_text_file(
    path: Path,
    include_extensions: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
) -> bool:
    """Check if a file is likely a text file based on extension or name.

    Args:
        path: Path to the file.
        include_extensions: If provided, only files with these extensions are accepted.
            Extensions should include the leading dot (e.g. {".py", ".js"}).
            Special filenames (e.g. "Dockerfile") are still accepted regardless.
        exclude_extensions: If provided, files with these extensions are rejected.
            Applied after include_extensions check.
    """
    suffix = path.suffix.lower()

    if include_extensions is not None:
        # Explicit allowlist: special filenames pass through, others must match
        if path.name not in TEXT_FILENAMES and suffix not in include_extensions:
            return False
        return not (exclude_extensions and suffix in exclude_extensions)

    # Default behaviour: must be a known text extension or special filename
    if path.name in TEXT_FILENAMES:
        return not (exclude_extensions and suffix in exclude_extensions)

    if suffix not in TEXT_EXTENSIONS:
        return False

    return not (exclude_extensions and suffix in exclude_extensions)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text]

        sep = seps[0] if seps else ""
        remaining = seps[1:] if len(seps) > 1 else [""]

        if sep == "":
            # Base case: hard split
            result = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                piece = text[i : i + chunk_size]
                if piece.strip():
                    result.append(piece)
            return result

        parts = text.split(sep)
        current_chunk = ""
        result = []

        for part in parts:
            candidate = current_chunk + sep + part if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    result.append(current_chunk)
                if len(part) > chunk_size:
                    result.extend(_split(part, remaining))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            result.append(current_chunk)
        return result

    chunks = _split(text, separators)

    # Apply overlap by including trailing context from previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + chunks[i])
        return overlapped

    return chunks if chunks else ([text] if text.strip() else [])


def _file_hash(path: Path) -> str:
    """Quick hash of file contents for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def needs_ingestion(directory: str) -> bool:
    """Check if directory is already indexed in Qdrant."""
    return not is_directory_indexed(directory)


def check_local_changes(directory: str) -> dict:
    """Use git status to detect uncommitted changes in the directory."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"is_git_repo": False, "has_changes": False, "details": "Not a git repository"}

        changes = result.stdout.strip()
        return {
            "is_git_repo": True,
            "has_changes": bool(changes),
            "changed_files": len(changes.splitlines()) if changes else 0,
            "details": changes[:500] if changes else "No uncommitted changes",
        }
    except Exception as e:
        return {"is_git_repo": False, "has_changes": False, "details": str(e)}


def _get_current_commit(directory: str) -> str | None:
    """Get the current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_last_indexed_commit(directory: str) -> str | None:
    """Read the commit hash from the last ingestion marker."""
    marker = Path(directory) / ".codebase-rag" / "last_commit.txt"
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return None


def _save_indexed_commit(directory: str, commit_hash: str) -> None:
    """Save the commit hash after ingestion."""
    marker_dir = Path(directory) / ".codebase-rag"
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / "last_commit.txt").write_text(commit_hash, encoding="utf-8")


def get_changed_files(directory: str) -> list[str]:
    """Get files changed since last ingestion using git diff.

    Returns relative paths of files that have changed.
    """
    last_commit = _get_last_indexed_commit(directory)
    changed = set()

    # Changes since last indexed commit
    if last_commit:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", last_commit, "HEAD"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed.update(result.stdout.strip().splitlines())
        except Exception:
            pass

    # Uncommitted changes (staged + unstaged)
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                # Format: "XY filename" or "XY old -> new" for renames
                parts = line[3:].split(" -> ")
                changed.add(parts[-1])
    except Exception:
        pass

    return list(changed)


def _collect_text_files(
    directory: str,
    include_extensions: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
) -> list[tuple[Path, str]]:
    """Walk directory and collect text files, respecting .gitignore.

    Args:
        directory: Root directory to walk.
        include_extensions: Normalised set of extensions to include (e.g. {".py"}).
            When provided, only files with these extensions are ingested.
        exclude_extensions: Normalised set of extensions to exclude (e.g. {".txt"}).
            Applied on top of the default TEXT_EXTENSIONS allowlist.
    """
    gitignore = _load_gitignore(directory)
    files = []
    base_path = Path(directory)

    for root, dirs, filenames in os.walk(directory):
        # Skip hidden and non-code directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]

        for filename in filenames:
            filepath = Path(root) / filename
            rel_path = str(filepath.relative_to(base_path))

            if gitignore and gitignore.match_file(rel_path):
                continue
            if not _is_text_file(filepath, include_extensions, exclude_extensions):
                continue
            if filepath.stat().st_size > 1_000_000:
                log.info("Skipping large file: %s", rel_path)
                continue

            files.append((filepath, rel_path))

    return files


def _embed_and_chunk_files(
    files: list[tuple[Path, str]],
    directory: str,
    timeout_seconds: float,
    start_time: float,
) -> tuple[list[dict], list[list[float]]]:
    """Chunk and embed a list of files. Returns (chunks, embeddings)."""
    all_chunks: list[dict] = []
    all_embeddings: list[list[float]] = []
    now = datetime.now(timezone.utc).isoformat()

    for filepath, rel_path in files:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            log.warning("Ingestion timeout reached after %d files", len(all_chunks))
            break

        try:
            text = filepath.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text, settings.chunk_size, settings.chunk_overlap)

            for i, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                all_chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "file_path": rel_path,
                        "directory": directory,
                        "chunk_index": i,
                        "ingested_at": now,
                    }
                )
                embedding = get_embedding(chunk_text)
                all_embeddings.append(embedding)

        except Exception as e:
            log.warning("Failed to process %s: %s", rel_path, e)
            continue

    return all_chunks, all_embeddings


def _normalise_extensions(extensions: list[str] | None) -> set[str] | None:
    """Normalise a list of extension strings to a lowercase set with leading dots.

    Accepts extensions with or without leading dot (e.g. "py" or ".py").
    Returns None if the input list is None or empty.
    """
    if not extensions:
        return None
    return {(ext if ext.startswith(".") else f".{ext}").lower() for ext in extensions}


def ingest_directory(
    directory: str,
    include_extensions: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
) -> str:
    """Walk directory, chunk files, embed, store in Qdrant.

    Respects .gitignore, skips binary files, uses recursive character splitting.
    Returns a status message.

    Args:
        directory: Root directory to ingest.
        include_extensions: If provided, only files with these extensions are
            ingested (e.g. [".py", ".ts"]). Overrides the default TEXT_EXTENSIONS.
        exclude_extensions: If provided, files with these extensions are skipped
            (e.g. [".txt", ".md"]).
    """
    timeout_seconds = settings.ingestion_timeout_hours * 3600
    start_time = time.time()

    directory = os.path.abspath(directory)
    log.info("Starting ingestion of %s (timeout: %dh)", directory, settings.ingestion_timeout_hours)

    inc_exts = _normalise_extensions(include_extensions)
    exc_exts = _normalise_extensions(exclude_extensions)

    # Ensure Qdrant collection exists
    vector_size = get_embedding_dimension()
    ensure_collection(vector_size)

    # Delete existing points for this directory (re-ingest cleanly)
    delete_directory_points(directory)

    files_to_ingest = _collect_text_files(directory, inc_exts, exc_exts)

    if not files_to_ingest:
        return f"No text files found to ingest in {directory}"

    log.info("Found %d text files to ingest", len(files_to_ingest))

    all_chunks, all_embeddings = _embed_and_chunk_files(
        files_to_ingest, directory, timeout_seconds, start_time
    )

    if not all_chunks:
        return f"No chunks generated from files in {directory}"

    # Upsert to Qdrant
    count = upsert_chunks(all_chunks, all_embeddings)
    elapsed = time.time() - start_time

    # Save commit marker for incremental indexing
    commit = _get_current_commit(directory)
    if commit:
        _save_indexed_commit(directory, commit)

    msg = (
        f"Ingested {count} chunks from {len(files_to_ingest)} files in {directory} ({elapsed:.1f}s)"
    )
    log.info(msg)
    return msg


def ingest_incremental(
    directory: str,
    include_extensions: list[str] | None = None,
    exclude_extensions: list[str] | None = None,
) -> str:
    """Incrementally update the index with only changed files.

    Uses git diff to detect changes since last ingestion. Only re-embeds
    changed files. Falls back to full ingestion if no prior commit marker.

    Args:
        directory: Root directory to ingest.
        include_extensions: If provided, only files with these extensions are
            ingested (e.g. [".py", ".ts"]). Overrides the default TEXT_EXTENSIONS.
        exclude_extensions: If provided, files with these extensions are skipped
            (e.g. [".txt", ".md"]).
    """
    directory = os.path.abspath(directory)
    last_commit = _get_last_indexed_commit(directory)

    if not last_commit:
        log.info("No prior ingestion marker — doing full ingestion")
        return ingest_directory(directory, include_extensions, exclude_extensions)

    changed_files = get_changed_files(directory)
    if not changed_files:
        return f"Index is up to date for {directory} (no changes since last ingestion)"

    timeout_seconds = settings.ingestion_timeout_hours * 3600
    start_time = time.time()

    log.info("Incremental ingestion: %d changed files in %s", len(changed_files), directory)

    inc_exts = _normalise_extensions(include_extensions)
    exc_exts = _normalise_extensions(exclude_extensions)

    # Ensure collection exists
    vector_size = get_embedding_dimension()
    ensure_collection(vector_size)

    # Filter to only text files that exist
    base_path = Path(directory)
    gitignore = _load_gitignore(directory)
    files_to_ingest = []

    for rel_path in changed_files:
        filepath = base_path / rel_path
        if not filepath.exists():
            # File was deleted — we'd need to remove its chunks
            # For now, skip (full re-ingest handles this)
            continue
        if gitignore and gitignore.match_file(rel_path):
            continue
        if not _is_text_file(filepath, inc_exts, exc_exts):
            continue
        if filepath.stat().st_size > 1_000_000:
            continue
        files_to_ingest.append((filepath, rel_path))

    if not files_to_ingest:
        return f"No indexable changes found in {directory}"

    # Delete old chunks for changed files before re-inserting
    for _filepath, rel_path in files_to_ingest:
        delete_file_points(rel_path)

    all_chunks, all_embeddings = _embed_and_chunk_files(
        files_to_ingest, directory, timeout_seconds, start_time
    )

    if not all_chunks:
        return f"No chunks generated from changed files in {directory}"

    count = upsert_chunks(all_chunks, all_embeddings)
    elapsed = time.time() - start_time

    # Update commit marker
    commit = _get_current_commit(directory)
    if commit:
        _save_indexed_commit(directory, commit)

    msg = (
        f"Incrementally indexed {count} chunks from {len(files_to_ingest)} changed files "
        f"in {directory} ({elapsed:.1f}s)"
    )
    log.info(msg)
    return msg
