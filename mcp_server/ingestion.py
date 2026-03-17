"""Auto-ingestion engine — walks a directory, chunks files, embeds, stores in Qdrant."""

import hashlib
import logging
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pathspec

from config.settings import settings
from mcp_server.embeddings import get_embedding, get_embedding_dimension
from mcp_server.qdrant_client import (
    delete_directory_points,
    ensure_collection,
    is_directory_indexed,
    upsert_chunks,
)

log = logging.getLogger("codebase-rag-mcp")

# File extensions considered as text (code, config, docs)
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".kt", ".kts",
    ".go", ".rs", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".swift", ".scala", ".sh", ".bash", ".zsh", ".fish",
    ".html", ".css", ".scss", ".less", ".vue", ".svelte",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".sql", ".graphql", ".proto",
    ".md", ".rst", ".txt", ".adoc",
    ".dockerfile", ".env.example",
    ".gitignore", ".editorconfig",
    ".gradle", ".cmake", ".makefile",
}

# Files without extensions that are typically text
TEXT_FILENAMES = {
    "Dockerfile", "Makefile", "CMakeLists.txt", "Jenkinsfile",
    "Procfile", "Vagrantfile", "Gemfile", "Rakefile",
    ".gitignore", ".dockerignore", ".editorconfig",
}


def _load_gitignore(directory: str) -> Optional[pathspec.PathSpec]:
    """Load .gitignore patterns from the directory."""
    gitignore_path = Path(directory) / ".gitignore"
    if not gitignore_path.exists():
        return None
    with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
        return pathspec.PathSpec.from_lines("gitignore", f)


def _is_text_file(path: Path) -> bool:
    """Check if a file is likely a text file based on extension or name."""
    if path.name in TEXT_FILENAMES:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", " ", ""]
    chunks = []

    def _split(text: str, separators: list[str]) -> list[str]:
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text]

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

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

        for part in parts:
            candidate = current_chunk + sep + part if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, remaining_seps))
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            chunks.append(current_chunk)

        return []

    _split(text, separators)

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


def ingest_directory(directory: str) -> str:
    """Walk directory, chunk files, embed, store in Qdrant.

    Respects .gitignore, skips binary files, uses recursive character splitting.
    Returns a status message.
    """
    timeout_seconds = settings.ingestion_timeout_hours * 3600
    start_time = time.time()

    directory = os.path.abspath(directory)
    log.info("Starting ingestion of %s (timeout: %dh)", directory, settings.ingestion_timeout_hours)

    # Ensure Qdrant collection exists
    vector_size = get_embedding_dimension()
    ensure_collection(vector_size)

    # Delete existing points for this directory (re-ingest cleanly)
    delete_directory_points(directory)

    # Load .gitignore
    gitignore = _load_gitignore(directory)

    # Walk directory and collect text files
    files_to_ingest = []
    base_path = Path(directory)
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-code directories
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in {"node_modules", "__pycache__", "venv", ".venv", "dist", "build", ".git"}
        ]

        for filename in files:
            filepath = Path(root) / filename
            rel_path = str(filepath.relative_to(base_path))

            # Skip gitignored files
            if gitignore and gitignore.match_file(rel_path):
                continue

            if not _is_text_file(filepath):
                continue

            # Skip very large files (> 1MB)
            if filepath.stat().st_size > 1_000_000:
                log.info("Skipping large file: %s", rel_path)
                continue

            files_to_ingest.append((filepath, rel_path))

    if not files_to_ingest:
        return f"No text files found to ingest in {directory}"

    log.info("Found %d text files to ingest", len(files_to_ingest))

    # Chunk and embed files
    all_chunks = []
    all_embeddings = []
    now = datetime.now(timezone.utc).isoformat()

    for filepath, rel_path in files_to_ingest:
        # Check timeout
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

    if not all_chunks:
        return f"No chunks generated from files in {directory}"

    # Upsert to Qdrant
    count = upsert_chunks(all_chunks, all_embeddings)
    elapsed = time.time() - start_time

    msg = (
        f"Ingested {count} chunks from {len(files_to_ingest)} files "
        f"in {directory} ({elapsed:.1f}s)"
    )
    log.info(msg)
    return msg
