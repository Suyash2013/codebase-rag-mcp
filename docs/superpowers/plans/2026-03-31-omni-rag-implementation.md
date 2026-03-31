# Omni-RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform codebase-rag-mcp (currently partially renamed to rag-mcp) into omni-rag-mcp — a general-purpose RAG MCP plugin with format-aware chunking, hybrid BM25+semantic search, an extractor plugin system, and a core/code tool split.

**Architecture:** Decompose the monolithic `ingestion.py` into extractor, chunker, and change-detection abstractions. Add BM25 alongside Qdrant for hybrid search with reciprocal rank fusion. Extractors and chunkers use registry patterns (matching the existing embedding provider pattern). Code-specific tools are conditionally registered.

**Tech Stack:** Python 3.10+, FastMCP, Qdrant, pydantic-settings, ONNX Runtime. No new external dependencies (BM25 is custom pure-Python).

**Spec:** `docs/superpowers/specs/2026-03-31-omni-rag-design.md`

**Branch:** `feature/general-purpose-rag` (building on existing Phase A bug fixes)

---

## File Structure

### New files to create:
```
mcp_server/extractors/__init__.py          # Auto-registers built-in extractors
mcp_server/extractors/base.py              # ExtractorBase ABC + registry
mcp_server/extractors/text.py              # Plain text fallback extractor
mcp_server/extractors/code.py              # Source code extractor (wraps current TEXT_EXTENSIONS logic)
mcp_server/extractors/markdown.py          # Markdown-aware extractor
mcp_server/extractors/structured_data.py   # JSON/YAML/CSV/TOML/XML extractor
mcp_server/extractors/pdf.py              # Optional PDF extractor
mcp_server/extractors/docx.py             # Optional DOCX extractor
mcp_server/extractors/image.py            # Optional image/OCR extractor
mcp_server/chunkers/__init__.py            # Auto-registers built-in chunkers
mcp_server/chunkers/base.py               # ChunkerBase ABC + registry
mcp_server/chunkers/recursive.py          # Current _chunk_text() extracted
mcp_server/chunkers/code_chunker.py       # AST/regex-boundary chunking
mcp_server/chunkers/markdown_chunker.py   # Heading-aware chunking
mcp_server/chunkers/structured_chunker.py # JSON keys / CSV rows / YAML sections
mcp_server/chunkers/paragraph_chunker.py  # Paragraph/section boundary chunking
mcp_server/change_detection/__init__.py    # Factory function
mcp_server/change_detection/base.py        # ChangeDetector ABC + ChangeReport
mcp_server/change_detection/git_detector.py    # Current git logic extracted
mcp_server/change_detection/file_hash_detector.py  # mtime+hash fallback
mcp_server/storage/__init__.py             # Re-exports
mcp_server/storage/bm25_index.py           # Custom BM25-Okapi + corpus persistence
mcp_server/storage/hybrid.py               # Reciprocal rank fusion
tests/test_extractors.py                   # Extractor tests
tests/test_chunkers.py                     # Chunker tests
tests/test_change_detection.py             # Change detection tests
tests/test_bm25.py                         # BM25 + hybrid search tests
```

### Existing files to modify:
```
mcp_server/ingestion.py                    # Slim down to orchestrator using new abstractions
mcp_server/qdrant_client.py               # Add content_type to payload, move to storage/ later
mcp_server/server.py                       # Conditional code plugin, tool rename
mcp_server/tools/search.py                # Wire hybrid search
mcp_server/tools/ingest.py                # Use change detection abstraction
mcp_server/tools/stats.py                 # Rename tool
mcp_server/tools/context.py               # Rename tool
mcp_server/tools/structure.py             # No rename, conditionally registered
mcp_server/migration.py                    # Add .rag-mcp -> .omni-rag migration
config/settings.py                         # New settings, prefix change
pyproject.toml                             # Package rename, optional deps
CLAUDE.md                                  # Update docs
```

---

## Phase 1: Extract Abstractions (no rename, tests keep passing)

### Task 1: Create Chunker Base + Extract RecursiveChunker

**Files:**
- Create: `mcp_server/chunkers/__init__.py`
- Create: `mcp_server/chunkers/base.py`
- Create: `mcp_server/chunkers/recursive.py`
- Create: `tests/test_chunkers.py`
- Modify: `mcp_server/ingestion.py`

- [ ] **Step 1: Create chunker base module with ABC and registry**

```python
# mcp_server/chunkers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single chunk of content ready for embedding."""
    text: str
    metadata: dict = field(default_factory=dict)


class ChunkerBase(ABC):
    """Interface for content chunkers."""

    @abstractmethod
    def content_types(self) -> set[str]:
        """Which content_types this chunker handles."""
        ...

    @abstractmethod
    def chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        """Split text into chunks."""
        ...


_chunker_registry: dict[str, ChunkerBase] = {}


def register_chunker(chunker: ChunkerBase) -> None:
    for ct in chunker.content_types():
        _chunker_registry[ct] = chunker


def get_chunker(content_type: str) -> ChunkerBase:
    return _chunker_registry.get(content_type, _chunker_registry.get("plain_text"))
```

- [ ] **Step 2: Write failing test for RecursiveChunker**

```python
# tests/test_chunkers.py
from mcp_server.chunkers.recursive import RecursiveChunker
from mcp_server.chunkers.base import Chunk


class TestRecursiveChunker:
    def setup_method(self):
        self.chunker = RecursiveChunker()

    def test_content_types(self):
        assert "plain_text" in self.chunker.content_types()

    def test_small_text_single_chunk(self):
        chunks = self.chunker.chunk("hello world", chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert isinstance(chunks[0], Chunk)

    def test_empty_text(self):
        chunks = self.chunker.chunk("", chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 0

    def test_splits_long_text(self):
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        chunks = self.chunker.chunk(text, chunk_size=20, chunk_overlap=5)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_metadata_passed_through(self):
        chunks = self.chunker.chunk(
            "hello", chunk_size=1000, chunk_overlap=0,
            metadata={"file_path": "test.txt"}
        )
        assert chunks[0].metadata["file_path"] == "test.txt"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Extract RecursiveChunker from ingestion.py**

Copy the `_chunk_text()` logic from `mcp_server/ingestion.py:84-144` into a new `RecursiveChunker` class:

```python
# mcp_server/chunkers/recursive.py
from mcp_server.chunkers.base import ChunkerBase, Chunk


class RecursiveChunker(ChunkerBase):
    """Recursive character text splitter. Default fallback chunker."""

    SEPARATORS = ["\n\n", "\n", " ", ""]

    def content_types(self) -> set[str]:
        return {"plain_text"}

    def chunk(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: dict | None = None,
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []
        raw_chunks = self._split(text, chunk_size, self.SEPARATORS)
        # Apply overlap
        result = []
        for i, chunk_text in enumerate(raw_chunks):
            if i > 0 and chunk_overlap > 0:
                prev = raw_chunks[i - 1]
                overlap_text = prev[-chunk_overlap:]
                chunk_text = overlap_text + chunk_text
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = i
            result.append(Chunk(text=chunk_text, metadata=meta))
        return result

    def _split(self, text: str, chunk_size: int, separators: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else ""
        remaining_seps = separators[1:] if len(separators) > 1 else [""]

        if sep == "":
            # Hard split at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size):
                piece = text[i : i + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            return chunks

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > chunk_size:
                    chunks.extend(self._split(part, chunk_size, remaining_seps))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks
```

- [ ] **Step 5: Create chunkers __init__.py**

```python
# mcp_server/chunkers/__init__.py
from mcp_server.chunkers.base import (
    Chunk,
    ChunkerBase,
    get_chunker,
    register_chunker,
)
from mcp_server.chunkers.recursive import RecursiveChunker

# Register built-in chunkers
register_chunker(RecursiveChunker())

__all__ = ["Chunk", "ChunkerBase", "get_chunker", "register_chunker", "RecursiveChunker"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: All PASS

- [ ] **Step 7: Update ingestion.py to use RecursiveChunker**

Replace the `_chunk_text()` call in `mcp_server/ingestion.py` with the new chunker. The `_chunk_text` function (lines 84-144) should be replaced with an import, and the call site in `_embed_and_chunk_files` (~line 320) updated:

```python
# At top of ingestion.py, add:
from mcp_server.chunkers import get_chunker

# Replace the _chunk_text() call in _embed_and_chunk_files with:
chunker = get_chunker("plain_text")
chunks_list = chunker.chunk(content, settings.chunk_size, settings.chunk_overlap,
                            metadata={"file_path": rel_path})
chunk_texts = [c.text for c in chunks_list]
```

Remove the old `_chunk_text()` function definition (lines 84-144).

- [ ] **Step 8: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 9: Commit**

```bash
git add mcp_server/chunkers/ tests/test_chunkers.py mcp_server/ingestion.py
git commit -m "refactor: extract RecursiveChunker from ingestion.py"
```

---

### Task 2: Create Extractor Base + TextExtractor + CodeExtractor

**Files:**
- Create: `mcp_server/extractors/__init__.py`
- Create: `mcp_server/extractors/base.py`
- Create: `mcp_server/extractors/text.py`
- Create: `mcp_server/extractors/code.py`
- Create: `tests/test_extractors.py`
- Modify: `mcp_server/ingestion.py`

- [ ] **Step 1: Create extractor base module**

```python
# mcp_server/extractors/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionResult:
    """Result of extracting content from a file."""
    text: str
    content_type: str  # "code", "markdown", "structured_data", "document", "plain_text"
    metadata: dict = field(default_factory=dict)


class ExtractorBase(ABC):
    """Interface for file content extractors."""

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        ...

    @abstractmethod
    def supported_filenames(self) -> set[str]:
        ...

    def can_extract(self, path: Path) -> bool:
        if path.name in self.supported_filenames():
            return True
        return path.suffix.lower() in self.supported_extensions()

    @abstractmethod
    def extract(self, path: Path) -> ExtractionResult:
        ...

    def max_file_size(self) -> int:
        return 1_000_000


_registry: dict[str, ExtractorBase] = {}
_fallback: ExtractorBase | None = None


def register_extractor(extractor: ExtractorBase, is_fallback: bool = False) -> None:
    global _fallback
    for ext in extractor.supported_extensions():
        _registry[ext] = extractor
    for name in extractor.supported_filenames():
        _registry[name] = extractor
    if is_fallback:
        _fallback = extractor


def get_extractor(path: Path) -> ExtractorBase | None:
    if path.name in _registry:
        return _registry[path.name]
    if path.suffix.lower() in _registry:
        return _registry[path.suffix.lower()]
    return _fallback
```

- [ ] **Step 2: Write failing tests for extractors**

```python
# tests/test_extractors.py
import os
from pathlib import Path
from mcp_server.extractors.base import ExtractionResult
from mcp_server.extractors.text import TextExtractor
from mcp_server.extractors.code import CodeExtractor


class TestTextExtractor:
    def setup_method(self):
        self.extractor = TextExtractor()

    def test_supported_extensions(self):
        exts = self.extractor.supported_extensions()
        assert ".txt" in exts
        assert ".log" in exts

    def test_extract_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = self.extractor.extract(f)
        assert isinstance(result, ExtractionResult)
        assert result.text == "hello world"
        assert result.content_type == "plain_text"

    def test_extract_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.text == ""
        assert result.content_type == "plain_text"


class TestCodeExtractor:
    def setup_method(self):
        self.extractor = CodeExtractor()

    def test_supported_extensions(self):
        exts = self.extractor.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".go" in exts

    def test_supported_filenames(self):
        names = self.extractor.supported_filenames()
        assert "Dockerfile" in names
        assert "Makefile" in names

    def test_extract_python_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "code"
        assert "def hello" in result.text
        assert result.metadata.get("language") == "python"

    def test_extract_js_file(self, tmp_path):
        f = tmp_path / "test.js"
        f.write_text("function hello() {}", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "code"
        assert result.metadata.get("language") == "javascript"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_extractors.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 4: Implement TextExtractor**

```python
# mcp_server/extractors/text.py
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult


class TextExtractor(ExtractorBase):
    """Fallback extractor for plain text files."""

    TEXT_EXTENSIONS = {".txt", ".log", ".env.example", ".gitignore", ".editorconfig"}

    def supported_extensions(self) -> set[str]:
        return self.TEXT_EXTENSIONS

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(text=text, content_type="plain_text")
```

- [ ] **Step 5: Implement CodeExtractor**

```python
# mcp_server/extractors/code.py
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult

# Language detection map
_LANG_MAP = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".less": "less",
    ".vue": "vue", ".svelte": "svelte",
    ".sql": "sql",
    ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf",
    ".dockerfile": "dockerfile",
    ".gradle": "gradle",
    ".cmake": "cmake",
    ".makefile": "makefile",
    ".r": "r", ".R": "r",
}

_CODE_EXTENSIONS = set(_LANG_MAP.keys())

_CODE_FILENAMES = {
    "Dockerfile", "Makefile", "CMakeLists.txt", "Jenkinsfile",
    "Procfile", "Vagrantfile", "Gemfile", "Rakefile",
    ".gitignore", ".dockerignore", ".eslintrc", ".prettierrc",
}

_FILENAME_LANG = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    "CMakeLists.txt": "cmake",
    "Jenkinsfile": "groovy",
    "Gemfile": "ruby",
    "Rakefile": "ruby",
}


class CodeExtractor(ExtractorBase):
    """Extractor for source code files."""

    def supported_extensions(self) -> set[str]:
        return _CODE_EXTENSIONS

    def supported_filenames(self) -> set[str]:
        return _CODE_FILENAMES

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        language = _FILENAME_LANG.get(path.name) or _LANG_MAP.get(path.suffix.lower(), "unknown")
        return ExtractionResult(
            text=text,
            content_type="code",
            metadata={"language": language},
        )
```

- [ ] **Step 6: Create extractors __init__.py**

```python
# mcp_server/extractors/__init__.py
from mcp_server.extractors.base import (
    ExtractionResult,
    ExtractorBase,
    get_extractor,
    register_extractor,
)
from mcp_server.extractors.text import TextExtractor
from mcp_server.extractors.code import CodeExtractor

# Register built-in extractors (order matters: last registered wins for overlapping extensions)
_text = TextExtractor()
_code = CodeExtractor()
register_extractor(_text, is_fallback=True)
register_extractor(_code)

__all__ = [
    "ExtractionResult", "ExtractorBase", "get_extractor", "register_extractor",
    "TextExtractor", "CodeExtractor",
]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_extractors.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add mcp_server/extractors/ tests/test_extractors.py
git commit -m "feat: add extractor plugin system with TextExtractor and CodeExtractor"
```

---

### Task 3: Extract Change Detection Abstraction

**Files:**
- Create: `mcp_server/change_detection/__init__.py`
- Create: `mcp_server/change_detection/base.py`
- Create: `mcp_server/change_detection/git_detector.py`
- Create: `mcp_server/change_detection/file_hash_detector.py`
- Create: `tests/test_change_detection.py`
- Modify: `mcp_server/ingestion.py`

- [ ] **Step 1: Create change detection base**

```python
# mcp_server/change_detection/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChangeReport:
    """Result of change detection."""
    has_changes: bool
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    details: str = ""


class ChangeDetector(ABC):
    """Interface for detecting file changes since last indexing."""

    @abstractmethod
    def detect_changes(self, directory: str) -> ChangeReport:
        ...

    @abstractmethod
    def save_checkpoint(self, directory: str) -> None:
        ...

    @abstractmethod
    def has_checkpoint(self, directory: str) -> bool:
        ...
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_change_detection.py
import json
import os
from pathlib import Path
from mcp_server.change_detection.base import ChangeReport
from mcp_server.change_detection.file_hash_detector import FileHashDetector


class TestFileHashDetector:
    def test_no_checkpoint_means_no_changes_detected(self, tmp_path):
        detector = FileHashDetector()
        assert not detector.has_checkpoint(str(tmp_path))

    def test_save_and_detect_no_changes(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        assert detector.has_checkpoint(str(tmp_path))
        report = detector.detect_changes(str(tmp_path))
        assert not report.has_changes
        assert report.changed_files == []
        assert report.deleted_files == []

    def test_detect_modified_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        f.write_text("world")
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "file.txt" in report.changed_files

    def test_detect_new_file(self, tmp_path):
        (tmp_path / "old.txt").write_text("old")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        (tmp_path / "new.txt").write_text("new")
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "new.txt" in report.changed_files

    def test_detect_deleted_file(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        detector = FileHashDetector()
        detector.save_checkpoint(str(tmp_path))
        f.unlink()
        report = detector.detect_changes(str(tmp_path))
        assert report.has_changes
        assert "file.txt" in report.deleted_files
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_change_detection.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 4: Implement GitDetector**

Extract the git functions from `mcp_server/ingestion.py` (lines 161-257):

```python
# mcp_server/change_detection/git_detector.py
import logging
import subprocess
from pathlib import Path
from mcp_server.change_detection.base import ChangeDetector, ChangeReport

log = logging.getLogger("rag-mcp")

DATA_DIR = ".rag-mcp"


class GitDetector(ChangeDetector):
    """Git-based change detection."""

    def detect_changes(self, directory: str) -> ChangeReport:
        last_commit = self._get_last_indexed_commit(directory)
        if not last_commit:
            return ChangeReport(has_changes=True, details="No prior index found")

        current_commit = self._get_current_commit(directory)
        changed = []
        deleted = []

        # Changes since last indexed commit
        if current_commit and current_commit != last_commit:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", last_commit, current_commit],
                    capture_output=True, text=True, cwd=directory, timeout=30
                )
                if result.returncode == 0:
                    for f in result.stdout.strip().split("\n"):
                        f = f.strip()
                        if f:
                            if (Path(directory) / f).exists():
                                changed.append(f)
                            else:
                                deleted.append(f)
            except Exception as e:
                log.warning("git diff failed: %s", e)

        # Uncommitted changes
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=directory, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if len(line) > 3:
                        status = line[:2]
                        filepath = line[3:].strip()
                        if status.strip() == "D":
                            if filepath not in deleted:
                                deleted.append(filepath)
                        elif filepath not in changed:
                            changed.append(filepath)
        except Exception as e:
            log.warning("git status failed: %s", e)

        has_changes = bool(changed or deleted)
        return ChangeReport(
            has_changes=has_changes,
            changed_files=changed,
            deleted_files=deleted,
            details=f"{len(changed)} changed, {len(deleted)} deleted since {last_commit[:8]}"
        )

    def save_checkpoint(self, directory: str) -> None:
        commit = self._get_current_commit(directory)
        if commit:
            marker = Path(directory) / DATA_DIR / "last_commit.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(commit)

    def has_checkpoint(self, directory: str) -> bool:
        return self._get_last_indexed_commit(directory) is not None

    def _get_current_commit(self, directory: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=directory, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _get_last_indexed_commit(self, directory: str) -> str | None:
        marker = Path(directory) / DATA_DIR / "last_commit.txt"
        if marker.exists():
            return marker.read_text().strip()
        return None
```

- [ ] **Step 5: Implement FileHashDetector**

```python
# mcp_server/change_detection/file_hash_detector.py
import hashlib
import json
import logging
import os
from pathlib import Path
from mcp_server.change_detection.base import ChangeDetector, ChangeReport

log = logging.getLogger("rag-mcp")

DATA_DIR = ".rag-mcp"
MANIFEST_FILE = "file_manifest.json"


class FileHashDetector(ChangeDetector):
    """File modification time + hash based change detection for non-git directories."""

    def detect_changes(self, directory: str) -> ChangeReport:
        manifest_path = Path(directory) / DATA_DIR / MANIFEST_FILE
        if not manifest_path.exists():
            return ChangeReport(has_changes=True, details="No prior manifest found")

        old_manifest = json.loads(manifest_path.read_text())
        current_manifest = self._build_manifest(directory)

        changed = []
        deleted = []

        # Check for modified or new files
        for rel_path, info in current_manifest.items():
            if rel_path not in old_manifest:
                changed.append(rel_path)
            elif info["hash"] != old_manifest[rel_path]["hash"]:
                changed.append(rel_path)

        # Check for deleted files
        for rel_path in old_manifest:
            if rel_path not in current_manifest:
                deleted.append(rel_path)

        has_changes = bool(changed or deleted)
        return ChangeReport(
            has_changes=has_changes,
            changed_files=changed,
            deleted_files=deleted,
            details=f"{len(changed)} changed, {len(deleted)} deleted"
        )

    def save_checkpoint(self, directory: str) -> None:
        manifest = self._build_manifest(directory)
        manifest_path = Path(directory) / DATA_DIR / MANIFEST_FILE
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def has_checkpoint(self, directory: str) -> bool:
        return (Path(directory) / DATA_DIR / MANIFEST_FILE).exists()

    def _build_manifest(self, directory: str) -> dict:
        manifest = {}
        dir_path = Path(directory)
        for root, dirs, files in os.walk(directory):
            # Skip hidden and data dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                fpath = Path(root) / fname
                try:
                    rel = str(fpath.relative_to(dir_path)).replace("\\", "/")
                    stat = fpath.stat()
                    file_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
                    manifest[rel] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "hash": file_hash,
                    }
                except (OSError, PermissionError):
                    continue
        return manifest
```

- [ ] **Step 6: Create change_detection __init__.py with factory**

```python
# mcp_server/change_detection/__init__.py
from pathlib import Path
from mcp_server.change_detection.base import ChangeDetector, ChangeReport
from mcp_server.change_detection.git_detector import GitDetector
from mcp_server.change_detection.file_hash_detector import FileHashDetector


def create_detector(directory: str) -> ChangeDetector:
    """Factory: returns GitDetector if in a git repo, else FileHashDetector."""
    if (Path(directory) / ".git").exists():
        return GitDetector()
    return FileHashDetector()


__all__ = ["ChangeDetector", "ChangeReport", "GitDetector", "FileHashDetector", "create_detector"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_change_detection.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add mcp_server/change_detection/ tests/test_change_detection.py
git commit -m "feat: add change detection abstraction with GitDetector and FileHashDetector"
```

---

### Task 4: Refactor ingestion.py to Use New Abstractions

**Files:**
- Modify: `mcp_server/ingestion.py`
- Modify: `mcp_server/tools/ingest.py`

- [ ] **Step 1: Refactor ingestion.py to use extractors and chunkers**

Replace file-reading logic in `_embed_and_chunk_files()` with extractor + chunker pipeline. Replace `_is_text_file()` and `_collect_text_files()` with extractor-based file collection. Replace git functions with change detection abstraction.

The key changes to `ingestion.py`:
1. Remove `_chunk_text()` (already done in Task 1)
2. Remove `_is_text_file()`, `TEXT_FILENAMES` — replaced by extractors
3. Remove `check_local_changes()`, `_get_current_commit()`, `_get_last_indexed_commit()`, `_save_indexed_commit()`, `get_changed_files()` — replaced by change_detection
4. Replace `_collect_text_files()` to use extractors for file eligibility
5. Update `_embed_and_chunk_files()` to use extractor → chunker pipeline
6. Update `ingest_directory()` and `ingest_incremental()` to use `create_detector()`

The ingestion.py should import from the new modules:
```python
from mcp_server.extractors import get_extractor
from mcp_server.chunkers import get_chunker
from mcp_server.change_detection import create_detector
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS (some tests may need minor import adjustments if they tested removed functions directly)

- [ ] **Step 3: Fix any broken tests**

Update test imports if they referenced removed private functions like `_chunk_text` or `_is_text_file`. These are now tested through their new module tests.

- [ ] **Step 4: Run full test suite again**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mcp_server/ingestion.py mcp_server/tools/ingest.py tests/
git commit -m "refactor: wire ingestion.py to use extractor/chunker/change-detection abstractions"
```

---

## Phase 2: New Capabilities

### Task 5: Add Format-Aware Chunkers

**Files:**
- Create: `mcp_server/chunkers/code_chunker.py`
- Create: `mcp_server/chunkers/markdown_chunker.py`
- Create: `mcp_server/chunkers/structured_chunker.py`
- Create: `mcp_server/chunkers/paragraph_chunker.py`
- Modify: `mcp_server/chunkers/__init__.py`
- Modify: `tests/test_chunkers.py`

- [ ] **Step 1: Write failing tests for all new chunkers**

Add to `tests/test_chunkers.py`:

```python
from mcp_server.chunkers.code_chunker import CodeChunker
from mcp_server.chunkers.markdown_chunker import MarkdownChunker
from mcp_server.chunkers.structured_chunker import StructuredChunker
from mcp_server.chunkers.paragraph_chunker import ParagraphChunker


class TestCodeChunker:
    def setup_method(self):
        self.chunker = CodeChunker()

    def test_content_types(self):
        assert "code" in self.chunker.content_types()

    def test_splits_python_at_function_boundaries(self):
        code = '''def foo():
    return 1

def bar():
    return 2

def baz():
    return 3
'''
        chunks = self.chunker.chunk(code, chunk_size=30, chunk_overlap=0,
                                     metadata={"language": "python"})
        # Should split at function boundaries, not mid-function
        assert len(chunks) >= 2
        assert any("def foo" in c.text for c in chunks)
        assert any("def bar" in c.text for c in chunks)

    def test_falls_back_for_unknown_language(self):
        code = "some random code\n" * 100
        chunks = self.chunker.chunk(code, chunk_size=50, chunk_overlap=0,
                                     metadata={"language": "unknown"})
        assert len(chunks) > 1


class TestMarkdownChunker:
    def setup_method(self):
        self.chunker = MarkdownChunker()

    def test_content_types(self):
        assert "markdown" in self.chunker.content_types()

    def test_splits_at_headings(self):
        md = "# Section 1\n\nContent one.\n\n# Section 2\n\nContent two."
        chunks = self.chunker.chunk(md, chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 2
        assert "Section 1" in chunks[0].text
        assert "Section 2" in chunks[1].text

    def test_preserves_heading_in_metadata(self):
        md = "# My Title\n\nSome content here."
        chunks = self.chunker.chunk(md, chunk_size=1000, chunk_overlap=0)
        assert chunks[0].metadata.get("section_title") == "My Title"

    def test_sub_splits_oversized_section(self):
        md = "# Big Section\n\n" + "word " * 500
        chunks = self.chunker.chunk(md, chunk_size=100, chunk_overlap=0)
        assert len(chunks) > 1


class TestStructuredChunker:
    def setup_method(self):
        self.chunker = StructuredChunker()

    def test_content_types(self):
        assert "structured_data" in self.chunker.content_types()

    def test_json_splits_at_top_level_keys(self):
        import json
        data = json.dumps({"key1": "value1" * 50, "key2": "value2" * 50})
        chunks = self.chunker.chunk(data, chunk_size=100, chunk_overlap=0,
                                     metadata={"format": "json"})
        assert len(chunks) >= 2

    def test_csv_preserves_header(self):
        csv = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF"
        chunks = self.chunker.chunk(csv, chunk_size=40, chunk_overlap=0,
                                     metadata={"format": "csv"})
        # Each chunk should include the header
        for c in chunks:
            assert "name,age,city" in c.text


class TestParagraphChunker:
    def setup_method(self):
        self.chunker = ParagraphChunker()

    def test_content_types(self):
        assert "document" in self.chunker.content_types()

    def test_splits_at_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = self.chunker.chunk(text, chunk_size=30, chunk_overlap=0)
        assert len(chunks) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chunkers.py -v -k "Code or Markdown or Structured or Paragraph"`
Expected: FAIL (imports not found)

- [ ] **Step 3: Implement CodeChunker**

```python
# mcp_server/chunkers/code_chunker.py
import ast
import re
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class CodeChunker(ChunkerBase):
    """Splits code at function/class boundaries when possible."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"code"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []
        lang = (metadata or {}).get("language", "unknown")

        if lang == "python":
            blocks = self._split_python(text)
        elif lang in ("javascript", "typescript", "go", "java", "kotlin"):
            blocks = self._split_by_regex(text, lang)
        else:
            return self._fallback.chunk(text, chunk_size, chunk_overlap, metadata)

        return self._blocks_to_chunks(blocks, chunk_size, chunk_overlap, metadata)

    def _split_python(self, text: str) -> list[str]:
        """Split Python code at top-level function/class boundaries using AST."""
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [text]

        lines = text.split("\n")
        blocks = []
        prev_end = 0

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                # Capture any code before this definition
                if start > prev_end:
                    preamble = "\n".join(lines[prev_end:start]).strip()
                    if preamble:
                        blocks.append(preamble)
                end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
                blocks.append("\n".join(lines[start:end]))
                prev_end = end

        # Remaining lines
        if prev_end < len(lines):
            remainder = "\n".join(lines[prev_end:]).strip()
            if remainder:
                blocks.append(remainder)

        return blocks if blocks else [text]

    def _split_by_regex(self, text: str, lang: str) -> list[str]:
        """Split code at function/class boundaries using regex."""
        patterns = {
            "javascript": r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\()",
            "typescript": r"^(?:export\s+)?(?:async\s+)?(?:function\s+\w+|class\s+\w+|interface\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\()",
            "go": r"^(?:func\s+|type\s+\w+\s+(?:struct|interface))",
            "java": r"^(?:\s*(?:public|private|protected)\s+)?(?:static\s+)?(?:class|interface|(?:\w+\s+)+\w+\s*\()",
            "kotlin": r"^(?:fun\s+|class\s+|interface\s+|object\s+)",
        }
        pattern = patterns.get(lang)
        if not pattern:
            return [text]

        lines = text.split("\n")
        blocks = []
        current_block: list[str] = []

        for line in lines:
            if re.match(pattern, line) and current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return [b for b in blocks if b.strip()]

    def _blocks_to_chunks(self, blocks: list[str], chunk_size: int,
                          chunk_overlap: int, metadata: dict | None) -> list[Chunk]:
        """Convert code blocks to chunks, merging small blocks and splitting large ones."""
        chunks = []
        current = ""

        for block in blocks:
            if len(current) + len(block) + 1 <= chunk_size:
                current = current + "\n" + block if current else block
            else:
                if current.strip():
                    meta = dict(metadata) if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Chunk(text=current.strip(), metadata=meta))
                if len(block) > chunk_size:
                    # Sub-split oversized blocks
                    sub_chunks = self._fallback.chunk(block, chunk_size, chunk_overlap, metadata)
                    for sc in sub_chunks:
                        sc.metadata["chunk_index"] = len(chunks)
                        chunks.append(sc)
                    current = ""
                else:
                    current = block

        if current.strip():
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=current.strip(), metadata=meta))

        return chunks
```

- [ ] **Step 4: Implement MarkdownChunker**

```python
# mcp_server/chunkers/markdown_chunker.py
import re
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class MarkdownChunker(ChunkerBase):
    """Splits markdown at heading boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"markdown"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        sections = self._split_by_headings(text)
        chunks = []

        for title, content in sections:
            section_text = content.strip()
            if not section_text:
                continue

            if len(section_text) <= chunk_size:
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                if title:
                    meta["section_title"] = title
                chunks.append(Chunk(text=section_text, metadata=meta))
            else:
                # Sub-split oversized sections
                sub_meta = dict(metadata) if metadata else {}
                if title:
                    sub_meta["section_title"] = title
                sub_chunks = self._fallback.chunk(section_text, chunk_size, chunk_overlap, sub_meta)
                for sc in sub_chunks:
                    sc.metadata["chunk_index"] = len(chunks)
                    chunks.append(sc)

        return chunks

    def _split_by_headings(self, text: str) -> list[tuple[str | None, str]]:
        """Split text into (heading_title, content) pairs."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = []
        last_end = 0
        last_title = None

        for match in heading_pattern.finditer(text):
            if match.start() > last_end:
                content = text[last_end:match.start()]
                if content.strip():
                    sections.append((last_title, content))
            last_title = match.group(2).strip()
            last_end = match.end()

        # Remaining content
        if last_end < len(text):
            remaining = text[last_end:]
            if remaining.strip():
                sections.append((last_title, remaining))
        elif last_title and not sections:
            sections.append((last_title, text))

        if not sections and text.strip():
            sections.append((None, text))

        return sections
```

- [ ] **Step 5: Implement StructuredChunker**

```python
# mcp_server/chunkers/structured_chunker.py
import json
import re
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class StructuredChunker(ChunkerBase):
    """Splits structured data (JSON, YAML, CSV) at logical boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"structured_data"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        fmt = (metadata or {}).get("format", "").lower()

        if fmt == "csv":
            return self._chunk_csv(text, chunk_size, metadata)
        elif fmt == "json":
            return self._chunk_json(text, chunk_size, metadata)
        else:
            return self._fallback.chunk(text, chunk_size, chunk_overlap, metadata)

    def _chunk_csv(self, text: str, chunk_size: int, metadata: dict | None) -> list[Chunk]:
        """Split CSV by row groups, preserving header in each chunk."""
        lines = text.strip().split("\n")
        if not lines:
            return []

        header = lines[0]
        rows = lines[1:]
        chunks = []
        current_rows: list[str] = []
        current_size = len(header)

        for row in rows:
            if current_size + len(row) + 1 > chunk_size and current_rows:
                chunk_text = header + "\n" + "\n".join(current_rows)
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                chunks.append(Chunk(text=chunk_text, metadata=meta))
                current_rows = []
                current_size = len(header)
            current_rows.append(row)
            current_size += len(row) + 1

        if current_rows:
            chunk_text = header + "\n" + "\n".join(current_rows)
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=chunk_text, metadata=meta))

        return chunks

    def _chunk_json(self, text: str, chunk_size: int, metadata: dict | None) -> list[Chunk]:
        """Split JSON at top-level keys."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self._fallback.chunk(text, chunk_size, 0, metadata)

        if not isinstance(data, dict):
            return self._fallback.chunk(text, chunk_size, 0, metadata)

        chunks = []
        current_obj: dict = {}
        current_size = 2  # {}

        for key, value in data.items():
            entry_str = json.dumps({key: value})
            entry_size = len(entry_str)

            if current_size + entry_size > chunk_size and current_obj:
                meta = dict(metadata) if metadata else {}
                meta["chunk_index"] = len(chunks)
                chunks.append(Chunk(text=json.dumps(current_obj, indent=2), metadata=meta))
                current_obj = {}
                current_size = 2

            current_obj[key] = value
            current_size += entry_size

        if current_obj:
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=json.dumps(current_obj, indent=2), metadata=meta))

        return chunks
```

- [ ] **Step 6: Implement ParagraphChunker**

```python
# mcp_server/chunkers/paragraph_chunker.py
from mcp_server.chunkers.base import ChunkerBase, Chunk
from mcp_server.chunkers.recursive import RecursiveChunker


class ParagraphChunker(ChunkerBase):
    """Splits document content at paragraph boundaries."""

    _fallback = RecursiveChunker()

    def content_types(self) -> set[str]:
        return {"document"}

    def chunk(self, text: str, chunk_size: int, chunk_overlap: int,
              metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            candidate = current + "\n\n" + para if current else para
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    meta = dict(metadata) if metadata else {}
                    meta["chunk_index"] = len(chunks)
                    chunks.append(Chunk(text=current, metadata=meta))
                if len(para) > chunk_size:
                    sub_chunks = self._fallback.chunk(para, chunk_size, chunk_overlap, metadata)
                    for sc in sub_chunks:
                        sc.metadata["chunk_index"] = len(chunks)
                        chunks.append(sc)
                    current = ""
                else:
                    current = para

        if current:
            meta = dict(metadata) if metadata else {}
            meta["chunk_index"] = len(chunks)
            chunks.append(Chunk(text=current, metadata=meta))

        return chunks
```

- [ ] **Step 7: Update chunkers __init__.py to register all chunkers**

```python
# mcp_server/chunkers/__init__.py
from mcp_server.chunkers.base import Chunk, ChunkerBase, get_chunker, register_chunker
from mcp_server.chunkers.recursive import RecursiveChunker
from mcp_server.chunkers.code_chunker import CodeChunker
from mcp_server.chunkers.markdown_chunker import MarkdownChunker
from mcp_server.chunkers.structured_chunker import StructuredChunker
from mcp_server.chunkers.paragraph_chunker import ParagraphChunker

# Register built-in chunkers
register_chunker(RecursiveChunker())
register_chunker(CodeChunker())
register_chunker(MarkdownChunker())
register_chunker(StructuredChunker())
register_chunker(ParagraphChunker())

__all__ = [
    "Chunk", "ChunkerBase", "get_chunker", "register_chunker",
    "RecursiveChunker", "CodeChunker", "MarkdownChunker",
    "StructuredChunker", "ParagraphChunker",
]
```

- [ ] **Step 8: Run all chunker tests**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add mcp_server/chunkers/ tests/test_chunkers.py
git commit -m "feat: add format-aware chunkers (code, markdown, structured, paragraph)"
```

---

### Task 6: Add MarkdownExtractor and StructuredDataExtractor

**Files:**
- Create: `mcp_server/extractors/markdown.py`
- Create: `mcp_server/extractors/structured_data.py`
- Modify: `mcp_server/extractors/__init__.py`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_extractors.py`:

```python
from mcp_server.extractors.markdown import MarkdownExtractor
from mcp_server.extractors.structured_data import StructuredDataExtractor


class TestMarkdownExtractor:
    def setup_method(self):
        self.extractor = MarkdownExtractor()

    def test_supported_extensions(self):
        assert ".md" in self.extractor.supported_extensions()
        assert ".rst" in self.extractor.supported_extensions()

    def test_extract(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello\n\nWorld", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "markdown"
        assert "Hello" in result.text


class TestStructuredDataExtractor:
    def setup_method(self):
        self.extractor = StructuredDataExtractor()

    def test_supported_extensions(self):
        exts = self.extractor.supported_extensions()
        assert ".json" in exts
        assert ".yaml" in exts
        assert ".csv" in exts
        assert ".toml" in exts

    def test_extract_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}', encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "structured_data"
        assert result.metadata.get("format") == "json"

    def test_extract_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2", encoding="utf-8")
        result = self.extractor.extract(f)
        assert result.content_type == "structured_data"
        assert result.metadata.get("format") == "csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extractors.py -v -k "Markdown or Structured"`
Expected: FAIL

- [ ] **Step 3: Implement MarkdownExtractor**

```python
# mcp_server/extractors/markdown.py
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult


class MarkdownExtractor(ExtractorBase):
    """Extractor for markdown and similar document formats."""

    def supported_extensions(self) -> set[str]:
        return {".md", ".rst", ".adoc"}

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(
            text=text,
            content_type="markdown",
            metadata={"format": path.suffix.lstrip(".")},
        )
```

- [ ] **Step 4: Implement StructuredDataExtractor**

```python
# mcp_server/extractors/structured_data.py
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult

_FORMAT_MAP = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".csv": "csv",
    ".tsv": "csv",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
}


class StructuredDataExtractor(ExtractorBase):
    """Extractor for structured data formats."""

    def supported_extensions(self) -> set[str]:
        return set(_FORMAT_MAP.keys())

    def supported_filenames(self) -> set[str]:
        return set()

    def extract(self, path: Path) -> ExtractionResult:
        text = path.read_text(encoding="utf-8", errors="replace")
        fmt = _FORMAT_MAP.get(path.suffix.lower(), "unknown")
        return ExtractionResult(
            text=text,
            content_type="structured_data",
            metadata={"format": fmt},
        )
```

- [ ] **Step 5: Update extractors __init__.py**

Add registrations for MarkdownExtractor and StructuredDataExtractor:

```python
from mcp_server.extractors.markdown import MarkdownExtractor
from mcp_server.extractors.structured_data import StructuredDataExtractor

_markdown = MarkdownExtractor()
_structured = StructuredDataExtractor()
register_extractor(_markdown)
register_extractor(_structured)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_extractors.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add mcp_server/extractors/ tests/test_extractors.py
git commit -m "feat: add MarkdownExtractor and StructuredDataExtractor"
```

---

### Task 7: Implement BM25 + Hybrid Search

**Files:**
- Create: `mcp_server/storage/__init__.py`
- Create: `mcp_server/storage/bm25_index.py`
- Create: `mcp_server/storage/hybrid.py`
- Create: `tests/test_bm25.py`
- Modify: `mcp_server/qdrant_client.py`
- Modify: `mcp_server/tools/search.py`

- [ ] **Step 1: Write failing tests for BM25**

```python
# tests/test_bm25.py
import json
from pathlib import Path
from mcp_server.storage.bm25_index import BM25Index
from mcp_server.storage.hybrid import reciprocal_rank_fusion


class TestBM25Index:
    def test_build_and_search(self, tmp_path):
        index = BM25Index(str(tmp_path))
        chunks = [
            {"id": "1", "text": "python function decorator pattern"},
            {"id": "2", "text": "javascript async await promise"},
            {"id": "3", "text": "python class inheritance method"},
        ]
        index.build(chunks)
        results = index.search("python decorator", top_k=2)
        assert len(results) > 0
        ids = [r[0] for r in results]
        assert "1" in ids  # Most relevant

    def test_empty_index(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build([])
        results = index.search("anything", top_k=5)
        assert results == []

    def test_persistence(self, tmp_path):
        index = BM25Index(str(tmp_path))
        chunks = [
            {"id": "1", "text": "hello world"},
            {"id": "2", "text": "goodbye world"},
        ]
        index.build(chunks)
        index.save()

        # Load in new instance
        index2 = BM25Index(str(tmp_path))
        index2.load()
        results = index2.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0][0] == "1"

    def test_update_corpus(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build([{"id": "1", "text": "original content"}])
        index.save()
        index.update(
            add=[{"id": "2", "text": "new content"}],
            remove=["1"],
        )
        results = index.search("new content", top_k=5)
        ids = [r[0] for r in results]
        assert "2" in ids
        assert "1" not in ids


class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        semantic = [
            {"id": "a", "score": 0.9, "text": "a"},
            {"id": "b", "score": 0.8, "text": "b"},
            {"id": "c", "score": 0.7, "text": "c"},
        ]
        bm25 = [("b", 5.0), ("d", 4.0), ("a", 3.0)]
        fused = reciprocal_rank_fusion(semantic, bm25, k=60)
        # b appears in both lists, should rank high
        ids = [r["id"] for r in fused]
        assert "b" in ids[:2]

    def test_empty_bm25(self):
        semantic = [{"id": "a", "score": 0.9, "text": "a"}]
        fused = reciprocal_rank_fusion(semantic, [], k=60)
        assert len(fused) == 1
        assert fused[0]["id"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BM25Index**

```python
# mcp_server/storage/bm25_index.py
import json
import math
import re
import logging
from pathlib import Path

log = logging.getLogger("rag-mcp")

DATA_DIR = ".rag-mcp"


class BM25Index:
    """Lightweight BM25-Okapi implementation. No external dependencies."""

    def __init__(self, directory: str, k1: float = 1.5, b: float = 0.75):
        self.directory = directory
        self.k1 = k1
        self.b = b
        self._corpus_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks. Each chunk has 'id' and 'text'."""
        self._corpus_ids = [c["id"] for c in chunks]
        self._corpus_tokens = [self._tokenize(c["text"]) for c in chunks]
        self._compute_stats()

    def update(self, add: list[dict] | None = None, remove: list[str] | None = None) -> None:
        """Incrementally update the corpus."""
        if remove:
            remove_set = set(remove)
            pairs = [(cid, tokens) for cid, tokens in zip(self._corpus_ids, self._corpus_tokens)
                     if cid not in remove_set]
            if pairs:
                self._corpus_ids, self._corpus_tokens = zip(*pairs)
                self._corpus_ids = list(self._corpus_ids)
                self._corpus_tokens = list(self._corpus_tokens)
            else:
                self._corpus_ids = []
                self._corpus_tokens = []

        if add:
            for c in add:
                self._corpus_ids.append(c["id"])
                self._corpus_tokens.append(self._tokenize(c["text"]))

        self._compute_stats()

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs, sorted by score descending."""
        if not self._corpus_ids:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._corpus_tokens):
            score = 0.0
            doc_len = self._doc_lens[i]
            tf_map: dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = tf_map.get(qt, 0)
                idf = self._idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((self._corpus_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self) -> None:
        data_path = Path(self.directory) / DATA_DIR
        data_path.mkdir(parents=True, exist_ok=True)

        corpus = [{"id": cid, "tokens": tokens}
                  for cid, tokens in zip(self._corpus_ids, self._corpus_tokens)]
        (data_path / "bm25_corpus.json").write_text(json.dumps(corpus))

    def load(self) -> bool:
        corpus_path = Path(self.directory) / DATA_DIR / "bm25_corpus.json"
        if not corpus_path.exists():
            return False
        try:
            corpus = json.loads(corpus_path.read_text())
            self._corpus_ids = [c["id"] for c in corpus]
            self._corpus_tokens = [c["tokens"] for c in corpus]
            self._compute_stats()
            return True
        except Exception as e:
            log.warning("Failed to load BM25 index: %s", e)
            return False

    def _compute_stats(self) -> None:
        self._n_docs = len(self._corpus_ids)
        self._doc_lens = [len(t) for t in self._corpus_tokens]
        self._avg_dl = sum(self._doc_lens) / self._n_docs if self._n_docs else 0

        self._doc_freqs = {}
        for tokens in self._corpus_tokens:
            seen = set(tokens)
            for t in seen:
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1

        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        return re.findall(r"\w+", text.lower())
```

- [ ] **Step 4: Implement hybrid.py**

```python
# mcp_server/storage/hybrid.py


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """Fuse semantic and BM25 rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    # Semantic ranking
    for rank, result in enumerate(semantic_results):
        rid = result["id"] if "id" in result else result.get("file_path", str(rank))
        scores[rid] = scores.get(rid, 0) + semantic_weight / (k + rank + 1)
        result_map[rid] = result

    # BM25 ranking
    for rank, (chunk_id, bm25_score) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank + 1)
        if chunk_id not in result_map:
            result_map[chunk_id] = {"id": chunk_id}

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for rid in sorted_ids:
        entry = dict(result_map[rid])
        entry["fused_score"] = scores[rid]
        results.append(entry)

    return results
```

- [ ] **Step 5: Create storage __init__.py**

```python
# mcp_server/storage/__init__.py
from mcp_server.storage.bm25_index import BM25Index
from mcp_server.storage.hybrid import reciprocal_rank_fusion

__all__ = ["BM25Index", "reciprocal_rank_fusion"]
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: All PASS

- [ ] **Step 7: Wire BM25 into ingestion pipeline**

Modify `mcp_server/ingestion.py` to build/update BM25 index during ingestion:
- After chunks are created, feed them to `BM25Index.build()` or `.update()`
- Save the BM25 index after successful ingestion

- [ ] **Step 8: Wire hybrid search into tools/search.py**

Modify `mcp_server/tools/search.py`:
- Load BM25 index lazily on first search
- Run both Qdrant semantic search and BM25 search
- Fuse results with `reciprocal_rank_fusion()`
- Respect `settings.hybrid_search_enabled` config flag

- [ ] **Step 9: Add hybrid search settings to config**

Add to `config/settings.py`:
```python
hybrid_search_enabled: bool = True
hybrid_semantic_weight: float = 0.7
hybrid_bm25_weight: float = 0.3
```

- [ ] **Step 10: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add mcp_server/storage/ tests/test_bm25.py mcp_server/ingestion.py mcp_server/tools/search.py config/settings.py
git commit -m "feat: add BM25 + semantic hybrid search with reciprocal rank fusion"
```

---

### Task 8: Add Optional Extractors (PDF, DOCX, Image)

**Files:**
- Create: `mcp_server/extractors/pdf.py`
- Create: `mcp_server/extractors/docx.py`
- Create: `mcp_server/extractors/image.py`
- Modify: `mcp_server/extractors/__init__.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_extractors.py`

- [ ] **Step 1: Write tests for optional extractors**

```python
# Add to tests/test_extractors.py
import importlib


class TestOptionalExtractors:
    def test_pdf_extractor_importable(self):
        """PDF extractor should be importable even without pymupdf."""
        from mcp_server.extractors.pdf import PdfExtractor
        ext = PdfExtractor()
        assert ".pdf" in ext.supported_extensions()

    def test_docx_extractor_importable(self):
        from mcp_server.extractors.docx import DocxExtractor
        ext = DocxExtractor()
        assert ".docx" in ext.supported_extensions()

    def test_image_extractor_importable(self):
        from mcp_server.extractors.image import ImageExtractor
        ext = ImageExtractor()
        assert ".png" in ext.supported_extensions()
```

- [ ] **Step 2: Implement PDF extractor with graceful dependency handling**

```python
# mcp_server/extractors/pdf.py
import logging
from pathlib import Path
from mcp_server.extractors.base import ExtractorBase, ExtractionResult

log = logging.getLogger("rag-mcp")

try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class PdfExtractor(ExtractorBase):
    """PDF content extractor. Requires: pip install omni-rag-mcp[pdf]"""

    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    def supported_filenames(self) -> set[str]:
        return set()

    def can_extract(self, path: Path) -> bool:
        if not HAS_PYMUPDF:
            return False
        return super().can_extract(path)

    def extract(self, path: Path) -> ExtractionResult:
        if not HAS_PYMUPDF:
            return ExtractionResult(
                text="[PDF extraction requires pymupdf: pip install omni-rag-mcp[pdf]]",
                content_type="plain_text",
            )
        doc = pymupdf.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n\n".join(pages)
        return ExtractionResult(
            text=text,
            content_type="document",
            metadata={"format": "pdf", "pages": len(pages)},
        )

    def max_file_size(self) -> int:
        return 50_000_000  # 50MB for PDFs
```

- [ ] **Step 3: Implement DOCX and Image extractors** (similar pattern)

Follow the same graceful-import pattern for `docx.py` (using `python-docx`) and `image.py` (using `pytesseract` + `Pillow`).

- [ ] **Step 4: Add optional deps to pyproject.toml**

```toml
[project.optional-dependencies]
pdf = ["pymupdf>=1.24.0"]
docx = ["python-docx>=1.1.0"]
image = ["pytesseract>=0.3.10", "Pillow>=10.0"]
all = ["rag-mcp[pdf,docx,image]"]
```

- [ ] **Step 5: Register optional extractors in __init__.py**

Only register if `can_extract` returns True (dependency available):

```python
from mcp_server.extractors.pdf import PdfExtractor
from mcp_server.extractors.docx import DocxExtractor
from mcp_server.extractors.image import ImageExtractor

for _ext_cls in [PdfExtractor, DocxExtractor, ImageExtractor]:
    _ext = _ext_cls()
    if hasattr(_ext, '_has_deps') or True:  # Always register, can_extract gates usage
        register_extractor(_ext)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_extractors.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add mcp_server/extractors/ pyproject.toml tests/test_extractors.py
git commit -m "feat: add optional PDF, DOCX, and image extractors"
```

---

## Phase 3: Rename + Conditional Code Tools

### Task 9: Add content_type to Qdrant Payload

**Files:**
- Modify: `mcp_server/qdrant_client.py`
- Modify: `mcp_server/ingestion.py`

- [ ] **Step 1: Update upsert_chunks to accept content_type and section_title**

Modify `mcp_server/qdrant_client.py:upsert_chunks()` to include `content_type`, `section_title`, and `extractor` in the payload if present in the chunk dict.

- [ ] **Step 2: Update search to optionally filter by content_type**

Add `content_type_filter: str | None = None` parameter to `search()` in `qdrant_client.py`.

- [ ] **Step 3: Update ingestion to pass content_type through**

Ensure the extractor's `content_type` and chunk metadata flow through to Qdrant payload.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add mcp_server/qdrant_client.py mcp_server/ingestion.py
git commit -m "feat: add content_type to Qdrant payload for filtered search"
```

---

### Task 10: Rename Project to omni-rag

**Files:**
- Modify: `pyproject.toml` — name, CLI entry points
- Modify: `mcp_server/server.py` — FastMCP name, instructions
- Modify: `mcp_server/cli.py` — logger name, banner text
- Modify: `mcp_server/migration.py` — add .rag-mcp -> .omni-rag migration
- Modify: `config/settings.py` — prefix to OMNI_RAG_, paths to .omni-rag
- Modify: all files referencing `rag-mcp` or `.rag-mcp`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update pyproject.toml**

```toml
name = "omni-rag-mcp"
[project.scripts]
omni-rag = "mcp_server.cli:main"
omni-rag-setup = "mcp_server.cli:setup"
# Keep deprecated aliases
rag-mcp = "mcp_server.cli:main"
rag-mcp-setup = "mcp_server.cli:setup"
```

- [ ] **Step 2: Update config/settings.py**

Change `env_prefix` to `"OMNI_RAG_"`. Add migration shim to also read `RAG_` prefix vars. Update all `.rag-mcp` paths to `.omni-rag`.

- [ ] **Step 3: Update migration.py**

Add `.rag-mcp/` -> `.omni-rag/` migration alongside existing `.codebase-rag/` -> `.omni-rag/` migration. Add Qdrant collection auto-detection (try `"omni_rag"`, fall back to `"documents"`, fall back to `"codebase"`).

- [ ] **Step 4: Update server.py**

Change `FastMCP("rag-mcp", ...)` to `FastMCP("omni-rag", ...)`. Update instructions text.

- [ ] **Step 5: Update all logger names**

Change `logging.getLogger("rag-mcp")` to `logging.getLogger("omni-rag")` across all files.

- [ ] **Step 6: Rename tools with deprecated aliases**

In `server.py`, register new tool names and old names as aliases:

```python
# New names
mcp.tool()(search)
mcp.tool()(search_by_file)
# ... etc

# Deprecated aliases
@mcp.tool()
async def search_codebase(query: str, n_results: int = 0) -> str:
    """[Deprecated: use 'search'] Semantic search."""
    log.warning("search_codebase is deprecated, use search instead")
    return await search(query, n_results)
```

- [ ] **Step 7: Update skip_directories in settings**

Add `".omni-rag"` to the skip_directories list.

- [ ] **Step 8: Update CLAUDE.md**

Reflect the new project name, CLI commands, and config prefix.

- [ ] **Step 9: Add enable_code_plugin config and conditional registration**

Add `enable_code_plugin: bool = True` to settings. In `server.py`, wrap code tool registration:

```python
if settings.enable_code_plugin:
    mcp.tool()(get_file_signatures)
    mcp.tool()(get_dependency_graph)
```

- [ ] **Step 10: Run full test suite, fix broken imports/paths**

Run: `python -m pytest tests/ -v`
Fix any references to old paths, old collection names, old env prefix.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: rename project to omni-rag-mcp with OMNI_RAG_ prefix and deprecated aliases"
```

---

## Phase 4: Polish

### Task 11: Update Tests for New Components

**Files:**
- Modify: all test files

- [ ] **Step 1: Update test_config.py for new prefix and settings**

Add tests for `OMNI_RAG_` prefix, `RAG_` backwards compat, new settings (hybrid_search_enabled, enable_code_plugin, etc.).

- [ ] **Step 2: Update test_migration.py for .rag-mcp -> .omni-rag**

Add test for the new migration path.

- [ ] **Step 3: Add integration test for hybrid search**

Test that search returns results from both BM25 and semantic paths.

- [ ] **Step 4: Add test for extractor → chunker routing**

Test that a markdown file goes through MarkdownExtractor → MarkdownChunker.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: update tests for omni-rag rename and new components"
```

---

### Task 12: Final Verification and Cleanup

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run linter**

Run: `ruff check mcp_server/ config/ tests/`
Fix any issues.

- [ ] **Step 3: Verify CLI starts**

Run: `python -m mcp_server.cli` (or equivalent entry point)
Expected: MCP server starts without errors

- [ ] **Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup and lint fixes"
```

- [ ] **Step 5: Create PR**

Push branch and create PR to master.
