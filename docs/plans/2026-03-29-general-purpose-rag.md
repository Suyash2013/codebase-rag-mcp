# General-Purpose RAG MCP Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Complete A5 consolidation, delete the cleanup worktree, and implement all Phase B (B1–B7) to transform `codebase-rag-mcp` into a general-purpose `rag-mcp`.

**Architecture:** All work happens in the existing `feature/general-purpose-rag` worktree. Changes cascade from settings → core modules → tools → project metadata. TDD for each task.

**Tech Stack:** Python 3.10+, FastMCP, Qdrant, Pydantic-Settings, ONNX/Ollama/OpenAI/Voyage embeddings

---

### Task 0: Housekeeping — Delete Cleanup Worktree

**Step 1: Remove worktree**

```bash
git worktree remove .worktrees/cleanup --force
git branch -D cleanup/repo-gha-code-tests
```

**Step 2: Verify**

```bash
git worktree list
# Expected: only master + feature/general-purpose-rag
```

**Step 3: Commit** — N/A (no source changes)

---

### Task 1: Complete A5 — Consolidate Hardcoded SKIP_DIRS

**Files:**
- Modify: `mcp_server/tools/structure.py:13-26`
- Test: `tests/test_structure.py`

**Step 1: Write the failing test**

```python
# tests/test_structure.py — add test
def test_get_file_signatures_uses_settings_skip_directories(tmp_path, monkeypatch):
    """Verify that get_file_signatures respects settings.skip_directories, not a hardcoded set."""
    from config.settings import settings
    monkeypatch.setattr(settings, "skip_directories", ["custom_skip"])
    monkeypatch.setattr(settings, "working_directory", str(tmp_path))

    # Create a file inside the custom skip dir — should be skipped
    skip_dir = tmp_path / "custom_skip"
    skip_dir.mkdir()
    (skip_dir / "skipped.py").write_text("def skipped(): pass")

    # Create a file outside — should appear
    (tmp_path / "found.py").write_text("def found(): pass")

    result = get_file_signatures()
    assert "found" in result
    assert "skipped" not in result
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_structure.py::test_get_file_signatures_uses_settings_skip_directories -v`
Expected: FAIL (hardcoded SKIP_DIRS ignores monkeypatched settings)

**Step 3: Implement — remove hardcoded SKIP_DIRS**

In `mcp_server/tools/structure.py`:
- Delete lines 12–26 (the `SKIP_DIRS` constant)
- Line 51: replace `SKIP_DIRS` with `set(settings.skip_directories)`

```python
# Line 51 becomes:
skip_dirs = set(settings.skip_directories)
dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_structure.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add mcp_server/tools/structure.py tests/test_structure.py
git commit -m "fix(A5): consolidate SKIP_DIRS in tools/structure.py to use settings"
```

---

### Task 2: B1 — Rename Project

**Files:**
- Modify: `pyproject.toml` (name, scripts, description)
- Modify: `config/settings.py` (data dir paths `.codebase-rag` → `.rag-mcp`, collection default `"codebase"` → `"documents"`)
- Modify: `mcp_server/server.py` (FastMCP name, logger, instructions text)
- Modify: `mcp_server/cli.py` (setup function names/references)
- Modify: `mcp_server/ingestion.py` (logger name, `.codebase-rag` path refs)
- Modify: `mcp_server/analysis/overview.py` (logger name, `.codebase-rag` path refs)
- Modify: `mcp_server/analysis/structure.py` (logger name)
- Modify: `mcp_server/embeddings/onnx_local.py` (logger name)
- Modify: `mcp_server/embeddings/openai.py` (logger name)
- Modify: `mcp_server/embeddings/voyage.py` (logger name)
- Modify: `mcp_server/embeddings/ollama.py` (logger name)
- New: `mcp_server/migration.py` (data dir + collection rename logic)
- Test: `tests/test_migration.py`

**Step 1: Write migration tests**

```python
# tests/test_migration.py
def test_migrate_data_dir_renames_old_to_new(tmp_path):
    """If .codebase-rag/ exists and .rag-mcp/ doesn't, rename it."""
    old_dir = tmp_path / ".codebase-rag"
    old_dir.mkdir()
    (old_dir / "test.txt").write_text("data")

    migrate_data_directory(str(tmp_path))
    assert not old_dir.exists()
    assert (tmp_path / ".rag-mcp" / "test.txt").exists()

def test_migrate_data_dir_noop_when_new_exists(tmp_path):
    """If .rag-mcp/ already exists, do nothing."""
    (tmp_path / ".rag-mcp").mkdir()
    (tmp_path / ".codebase-rag").mkdir()

    migrate_data_directory(str(tmp_path))
    assert (tmp_path / ".codebase-rag").exists()  # old untouched
    assert (tmp_path / ".rag-mcp").exists()

def test_migrate_data_dir_noop_when_no_old(tmp_path):
    """If .codebase-rag/ doesn't exist, do nothing."""
    migrate_data_directory(str(tmp_path))
    assert not (tmp_path / ".rag-mcp").exists()
```

**Step 2: Run tests — expected FAIL (module doesn't exist)**

Run: `python -m pytest tests/test_migration.py -v`

**Step 3: Implement migration module**

Create `mcp_server/migration.py`:

```python
"""Data directory and collection migration from codebase-rag to rag-mcp."""
import logging
import os
from pathlib import Path

log = logging.getLogger("rag-mcp")

def migrate_data_directory(directory: str) -> None:
    old = Path(directory) / ".codebase-rag"
    new = Path(directory) / ".rag-mcp"
    if old.exists() and not new.exists():
        try:
            os.rename(str(old), str(new))
            log.info("Migrated data directory: %s -> %s", old, new)
        except OSError as e:
            log.warning(
                "Could not rename %s to %s: %s. Using old path with deprecation warning.",
                old, new, e,
            )
```

**Step 4: Rename all references across the project**

Global find-and-replace (carefully, file by file):
- Logger names: `"codebase-rag-mcp"` → `"rag-mcp"` in all files
- Data dir: `".codebase-rag"` → `".rag-mcp"` in `config/settings.py`, `overview.py`, `ingestion.py`
- FastMCP name: `"codebase-rag"` → `"rag-mcp"` in `server.py`
- Docstrings: `"codebase"` → `"documents"` for collection name default
- `pyproject.toml`: `name = "rag-mcp"`, scripts `rag-mcp` and `rag-mcp-setup`
- CLI references in `cli.py`: `"codebase-rag"` → `"rag-mcp"`
- Settings: `qdrant_collection: str = "documents"`, `qdrant_local_path` default to `.rag-mcp/qdrant`, `onnx_model_path` default to `.rag-mcp/models/`
- Also add `.rag-mcp` to `skip_directories` default list alongside `.codebase-rag` (keep old for compat)

**Step 5: Call migration at startup**

In `mcp_server/server.py` `__main__` block, call `migrate_data_directory()` before server starts. Also call it early in `ingestion.py:ingest_directory()`.

**Step 6: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: PASS (update test fixtures as needed for new paths/names)

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(B1): rename project to rag-mcp with backward-compatible migration"
```

---

### Task 3: B2 — Content-Agnostic File Type Detection

**Files:**
- Modify: `config/settings.py` (add `auto_detect_text: bool = True`)
- Modify: `mcp_server/ingestion.py` (`_is_text_file` to auto-detect)
- Test: `tests/test_ingestion.py`

**Step 1: Write failing test**

```python
def test_is_text_file_auto_detects_unknown_extension(tmp_path, monkeypatch):
    """Files with unknown extensions should be accepted if text-like."""
    from config.settings import settings
    monkeypatch.setattr(settings, "auto_detect_text", True)

    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25")
    assert _is_text_file(csv_file) is True

def test_is_text_file_rejects_binary(tmp_path, monkeypatch):
    """Binary files should be rejected even with auto-detect on."""
    from config.settings import settings
    monkeypatch.setattr(settings, "auto_detect_text", True)

    bin_file = tmp_path / "data.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03" * 100)
    assert _is_text_file(bin_file) is False
```

**Step 2: Run tests — expected FAIL**

Run: `python -m pytest tests/test_ingestion.py::test_is_text_file_auto_detects_unknown_extension -v`

**Step 3: Implement auto-detect**

Add to `config/settings.py`:
```python
auto_detect_text: bool = True
```

Update `_is_text_file()` in `ingestion.py` to add a final fallback when `auto_detect_text` is enabled: read first 8KB, reject if null bytes or high proportion of control characters, check UTF-8 decodability.

```python
def _is_binary_content(path: Path) -> bool:
    """Check if file content appears binary by reading first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return True
        control_chars = sum(1 for b in chunk if b < 0x09 or (0x0E <= b <= 0x1F))
        if len(chunk) > 0 and control_chars / len(chunk) > 0.1:
            return True
        chunk.decode("utf-8")
        return False
    except (UnicodeDecodeError, OSError):
        return True
```

Then at the end of `_is_text_file()`, before returning `False`:
```python
if settings.auto_detect_text and not _is_binary_content(path):
    return not (exclude_extensions and suffix in exclude_extensions)
```

**Step 4: Run tests — expected PASS**

Run: `python -m pytest tests/test_ingestion.py -v`

**Step 5: Commit**

```bash
git add config/settings.py mcp_server/ingestion.py tests/test_ingestion.py
git commit -m "feat(B2): content-agnostic text file detection with binary safety check"
```

---

### Task 4: B3 — Conditional Code-Specific Analysis

**Files:**
- Modify: `config/settings.py` (add `content_type: str = "auto"`)
- Modify: `mcp_server/analysis/overview.py` (add `is_code_project()` detection)
- Modify: `mcp_server/tools/structure.py` (gate on content type)
- Modify: `mcp_server/tools/context.py` (conditional formatting)
- Test: `tests/test_overview.py`, `tests/test_structure.py`

**Step 1: Write failing tests**

```python
# tests/test_overview.py
def test_is_code_project_detects_manifest(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")
    assert is_code_project(str(tmp_path)) is True

def test_is_code_project_false_for_docs(tmp_path):
    (tmp_path / "notes.md").write_text("# Notes")
    assert is_code_project(str(tmp_path)) is False

# tests/test_structure.py
def test_get_file_signatures_disabled_for_general_content(tmp_path, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "content_type", "general")
    monkeypatch.setattr(settings, "working_directory", str(tmp_path))
    (tmp_path / "notes.py").write_text("def hello(): pass")
    result = get_file_signatures()
    assert "only available for code projects" in result
```

**Step 2: Run tests — FAIL**

**Step 3: Implement**

Add to `config/settings.py`:
```python
content_type: str = "auto"  # "auto" | "code" | "general"
```

Add to `mcp_server/analysis/overview.py`:
```python
_MANIFEST_FILENAMES = set(_MANIFEST_FILES.keys())

def is_code_project(directory: str) -> bool:
    if settings.content_type == "code":
        return True
    if settings.content_type == "general":
        return False
    base = Path(directory)
    return any((base / m).exists() for m in _MANIFEST_FILENAMES)
```

Gate `get_file_signatures` and `get_dependency_graph` in `tools/structure.py`:
```python
from mcp_server.analysis.overview import is_code_project

def get_file_signatures(file_pattern: str = "") -> str:
    directory = settings.get_working_directory()
    if not is_code_project(directory):
        return ("Signature extraction is only available for code projects. "
                "Set RAG_CONTENT_TYPE=code to force enable.")
    # ... existing logic
```

**Step 4: Run tests — PASS**

Run: `python -m pytest tests/test_overview.py tests/test_structure.py -v`

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(B3): conditional code analysis based on RAG_CONTENT_TYPE"
```

---

### Task 5: B4 — Generalize Overview Analysis

**Files:**
- Modify: `mcp_server/analysis/overview.py` (`generate_overview` conditionals)
- Modify: `mcp_server/tools/context.py` (`_format_overview` general sections)
- Test: `tests/test_overview.py`

**Step 1: Write failing test**

```python
def test_generate_overview_general_has_file_type_distribution(tmp_path):
    """General directories should have file type distribution, not language breakdown."""
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "data.csv").write_text("a,b\n1,2")
    overview = generate_overview(str(tmp_path))
    assert "languages" in overview  # still present, generic
    assert "manifests" not in overview or overview["manifests"] == []
```

**Step 2: Implement**

In `generate_overview()`, wrap manifest/dependencies/key_files detection behind `is_code_project()` check. For general content, still include `languages` (which is really extension distribution) and `structure`.

In `_format_overview()`, change `"## Languages"` header to `"## File Types"` when not a code project.

**Step 3: Run tests — PASS**

Run: `python -m pytest tests/test_overview.py -v`

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(B4): generalize overview analysis with conditional code-specific sections"
```

---

### Task 6: B5 — Rename MCP Tools

**Files:**
- Modify: `mcp_server/tools/search.py` (rename functions)
- Modify: `mcp_server/tools/ingest.py` (rename function)
- Modify: `mcp_server/tools/context.py` (rename function)
- Modify: `mcp_server/server.py` (register new + old names as aliases)
- Test: `tests/test_tools.py` (if exists), manual verification

**Step 1: Rename functions and add aliases**

Rename table from spec:
| Old name | New name |
|----------|----------|
| `search_codebase` | `search` |
| `search_codebase_by_file` | `search_by_file` |
| `get_codebase_context` | `get_context` |
| `ingest_current_directory` | `ingest` |

For each: rename the function, then create an alias (e.g., `search_codebase = search`).

**Step 2: Update server.py registrations**

```python
# Register tools with new names
mcp.tool()(search)
mcp.tool()(search_by_file)
mcp.tool()(collection_stats)
mcp.tool()(ingest)
mcp.tool()(check_index_status)
mcp.tool()(get_context)
mcp.tool()(get_file_signatures)
mcp.tool()(get_dependency_graph)

# Backward-compatible aliases
mcp.tool(name="search_codebase")(search)
mcp.tool(name="search_codebase_by_file")(search_by_file)
mcp.tool(name="get_codebase_context")(get_context)
mcp.tool(name="ingest_current_directory")(ingest)
```

**Step 3: Update docstrings** to be content-agnostic ("search indexed files" not "search codebase").

**Step 4: Run all tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(B5): rename MCP tools with backward-compatible aliases"
```

---

### Task 7: B6 — Add Embedding Batching

**Files:**
- Modify: `mcp_server/embeddings/base.py` (add `embed_batch` abstract w/ default)
- Modify: `mcp_server/embeddings/__init__.py` (add `get_embedding_batch`)
- Modify: `mcp_server/embeddings/onnx_local.py` (native batch)
- Modify: `mcp_server/embeddings/openai.py` (API batch)
- Modify: `mcp_server/embeddings/voyage.py` (API batch)
- Modify: `mcp_server/embeddings/ollama.py` (loop fallback)
- Modify: `mcp_server/ingestion.py` (batch chunks before embedding)
- Test: `tests/test_embeddings.py`

**Step 1: Write failing test**

```python
def test_embed_batch_returns_list_of_vectors(mock_provider):
    """embed_batch should return one vector per input text."""
    texts = ["hello", "world", "test"]
    results = mock_provider.embed_batch(texts)
    assert len(results) == 3
    assert all(isinstance(v, list) for v in results)
```

**Step 2: Implement base class default**

```python
# base.py
def embed_batch(self, texts: list[str]) -> list[list[float]]:
    """Batch embed. Default: loop over embed(). Override for native batching."""
    return [self.embed(text) for text in texts]
```

**Step 3: Implement ONNX native batch**

Tokenize all texts at once, run single ONNX inference, mean-pool each.

**Step 4: Implement OpenAI/Voyage API batch**

Both APIs accept `"input": [list_of_texts]` natively.

**Step 5: Add `get_embedding_batch` to `__init__.py`**

```python
def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    return _get_provider().embed_batch(texts)
```

**Step 6: Update ingestion to use batching**

In `_embed_and_chunk_files()`, collect chunks in groups of 32, call `get_embedding_batch()` instead of per-chunk `get_embedding()`.

**Step 7: Run tests — PASS**

Run: `python -m pytest tests/test_embeddings.py tests/test_ingestion.py -v`

**Step 8: Commit**

```bash
git add -A
git commit -m "feat(B6): add embedding batching with native support for ONNX/OpenAI/Voyage"
```

---

### Task 8: B7 — Generalize Git Dependency

**Files:**
- New: `mcp_server/manifest.py` (file manifest for non-git change detection)
- Modify: `mcp_server/ingestion.py` (detect git vs. manifest at ingestion time)
- Test: `tests/test_manifest.py`

**Step 1: Write failing tests**

```python
# tests/test_manifest.py
def test_manifest_detects_new_files(tmp_path):
    manifest = FileManifest(str(tmp_path))
    (tmp_path / "a.txt").write_text("hello")
    changes = manifest.get_changes()
    assert "a.txt" in changes["added"]

def test_manifest_detects_modified_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    manifest = FileManifest(str(tmp_path))
    manifest.save()
    (tmp_path / "a.txt").write_text("world")
    changes = manifest.get_changes()
    assert "a.txt" in changes["modified"]

def test_manifest_detects_deleted_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    manifest = FileManifest(str(tmp_path))
    manifest.save()
    (tmp_path / "a.txt").unlink()
    changes = manifest.get_changes()
    assert "a.txt" in changes["deleted"]
```

**Step 2: Implement FileManifest**

```python
# mcp_server/manifest.py
class FileManifest:
    """Track file changes via mtime+size for non-git directories."""
    MANIFEST_FILE = ".rag-mcp/file_manifest.json"

    def __init__(self, directory: str): ...
    def save(self) -> None: ...
    def get_changes(self) -> dict[str, list[str]]: ...
```

**Step 3: Integrate into ingestion**

In `ingest_incremental()`:
- Check for `.git/` directory
- If git: use existing `get_changed_files()` (unchanged)
- If no git: use `FileManifest.get_changes()` to determine adds/modifies/deletes

**Step 4: Run tests — PASS**

Run: `python -m pytest tests/test_manifest.py tests/test_ingestion.py -v`

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(B7): manifest-based change detection for non-git directories"
```

---

## Verification Plan

### Automated Tests

After all tasks:

```bash
cd .worktrees/general-purpose-rag
python -m pytest tests/ -v --cov --cov-fail-under=65
```

Additional targeted test runs per task documented above.

### Integration Verification

1. **Non-code directory test:** Create a temp folder with `.md`, `.csv`, `.log` files (no git, no manifests). Run `ingest` tool, then `search`. Verify results return and code-specific tools show "not available" message.

2. **Code project test:** Point at the project itself. Verify `get_file_signatures`, `get_dependency_graph`, `get_context` all work as before.

3. **Old tool names:** Verify `search_codebase` still works as alias for `search`.

4. **Migration:** Create a `.codebase-rag/` directory in a temp folder, run the server initialization, verify rename to `.rag-mcp/`.

5. **Embedding batching:** Compare ingestion times before/after on a moderate directory (optional perf check).

### Manual Verification

- User to run `rag-mcp-setup` and verify Claude Code config is updated with new name.
- User to verify existing indexed projects still work after migration (no data loss).
