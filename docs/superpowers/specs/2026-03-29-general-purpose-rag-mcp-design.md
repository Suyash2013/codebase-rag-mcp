# General-Purpose RAG MCP: Bug Fixes & Generalization

**Date:** 2026-03-29
**Status:** Draft

## Context

`codebase-rag-mcp` is a Claude Code MCP plugin for semantic codebase search. It works well for code repositories but has several critical bugs and is tightly coupled to code-specific assumptions. The goal is to:

1. Fix existing bugs and harden the codebase (Phase A)
2. Generalize it to work with any directory of files, not just code (Phase B)
3. Rename the project to reflect its general-purpose nature
4. Keep code-specific analysis tools as optional, auto-detected features

## Approach

**Two-Phase Sequential**: Fix bugs first on the existing architecture, then generalize. This keeps PRs small and reviewable, and ensures generalization builds on a stable base.

---

## Phase A: Bug Fixes & Hardening

### A1. Fix duplicate chunks on incremental ingestion

**Problem:** `ingestion.py:541` — when a file changes, new chunks are added but old chunks are never deleted. This causes index bloat and stale search results.

**Fix:**
- Add `delete_file_points(collection, file_path)` to `mcp_server/qdrant_client.py` using Qdrant's `models.Filter` on the `file_path` payload field
- Call it in `_embed_and_chunk_files()` before upserting new chunks for each changed file
- Also handle deleted files: when `get_changed_files()` returns a file that no longer exists on disk, call `delete_file_points()` to clean up its stale chunks
- Files: `mcp_server/qdrant_client.py`, `mcp_server/ingestion.py`

### A2. Replace silent exception swallowing

**Problem:** Multiple `except Exception: pass` blocks in `overview.py` and `structure.py` hide real errors.

**Fix:**
- Replace `except Exception: pass` with `except Exception as e: log.warning("...: %s", e)`
- At ingestion/analysis completion, log aggregate error count
- Files: `mcp_server/analysis/overview.py`, `mcp_server/analysis/structure.py`

### A3. Fix ingestion timeout inconsistency

**Problem:** Timeout stops ingestion mid-way, leaving a partially-indexed state. `last_commit.txt` may or may not reflect what was actually indexed.

**Fix:**
- Only write `last_commit.txt` after fully successful ingestion
- Fix the deletion-before-re-indexing problem: `ingest_directory()` currently calls `delete_directory_points()` before embedding. Instead, use a two-phase approach — index new chunks first, then delete old ones (or use new UUIDs and delete stale points after success)
- On timeout, log which files were processed vs. skipped so the next run can continue
- Files: `mcp_server/ingestion.py`

### A4. Add cache invalidation for overview

**Problem:** `analysis/overview.py` caches to `overview.json` but never invalidates. Stale results persist across major project changes.

**Fix:**
- For git repos: use git HEAD commit hash as fingerprint (fast, no directory walk)
- For non-git dirs: compute a lightweight fingerprint (file count + total size of top-level entries) and store alongside the cache
- On `get_codebase_context`, compare fingerprint before returning cached data
- Files: `mcp_server/analysis/overview.py`

### A5. Make hardcoded limits configurable

**Problem:** File size limit (1MB), skip directories, and text extensions are hardcoded with no user override.

**Fix:** Add to `config/settings.py`:
- `max_file_size_bytes: int = 1_048_576` (1MB default)
- `skip_directories: list[str] = [...]` (current hardcoded list as default)
- `text_extensions: list[str] = [...]` (current TEXT_EXTENSIONS as default)

Update `ingestion.py` to read these from settings instead of module-level constants. Also consolidate the three independent `SKIP_DIRS` definitions (in `ingestion.py`, `overview.py`, and `structure.py`) into a single settings-driven source.

Files: `config/settings.py`, `mcp_server/ingestion.py`, `mcp_server/analysis/overview.py`, `mcp_server/analysis/structure.py`

### A6. Fix chunking implementation

**Problem:** `_chunk_text()` has a confusing closure pattern where `_split()` returns `[]` but accumulates chunks via outer scope. Overlap is applied post-hoc rather than during splitting.

**Fix:**
- First, add comprehensive edge-case tests for current chunking behavior (separator boundaries, overlap correctness, hard-split base case) to prevent regressions
- Refactor `_split()` to return chunks directly (no closure accumulation)
- Apply overlap during the splitting pass, not as a separate post-processing step
- Files: `mcp_server/ingestion.py`, `tests/test_ingestion.py`

### A7. Use Qdrant-native filtering for file patterns

**Problem:** `qdrant_client.py:151-179` fetches 5x results then filters client-side with substring matching. Wastes bandwidth and misses Qdrant's native capabilities.

**Fix:**
- Use Qdrant's `models.FieldCondition` with `models.MatchText` for substring filtering server-side
- Remove the over-fetch multiplier
- Files: `mcp_server/qdrant_client.py`

---

## Phase B: Generalization

### B1. Rename the project

**From → To:**
- Package: `codebase-rag-mcp` → `rag-mcp`
- CLI: `codebase-rag` → `rag-mcp`, `codebase-rag-setup` → `rag-mcp-setup`
- Logger: `codebase-rag-mcp` → `rag-mcp`
- Data dir: `.codebase-rag/` → `.rag-mcp/`
- Collection default: `"codebase"` → `"documents"`
- Settings prefix: keep `RAG_` (already generic)

**Files:** `pyproject.toml`, `mcp_server/cli.py`, `mcp_server/server.py`, `mcp_server/ingestion.py`, `config/settings.py`, `CLAUDE.md`, `README.md`, all test files

**Migration:**
- Data directory: if `.codebase-rag/` exists and `.rag-mcp/` doesn't, rename it automatically. On Windows, close the Qdrant client before renaming to avoid file locks; if rename fails, fall back to using the old path with a deprecation warning.
- Qdrant collection: if a collection named `"codebase"` exists but `"documents"` doesn't, rename it programmatically (or alias it). Do not silently lose existing indexed data.
- Files to update for `.codebase-rag` → `.rag-mcp` references: also `overview.py` and `structure.py` SKIP_DIRS sets.

### B2. Content-agnostic file type detection

**Current:** Hardcoded `TEXT_EXTENSIONS` set blocks non-listed file types.

**New behavior:**
- `RAG_TEXT_EXTENSIONS` setting (default: current set) — user can extend
- `RAG_AUTO_DETECT_TEXT` setting (default: true) — when enabled, files not in the extension list get a binary safety check: read first 8KB, reject if it contains null bytes or high proportion of control characters (0x00-0x08, 0x0E-0x1F excluding tab/newline/CR), then check if decodable as UTF-8
- Note: `text_extensions` from A5 is the same setting as `RAG_TEXT_EXTENSIONS` here — A5 creates it, B2 extends the behavior with auto-detection
- This lets users index directories with `.log`, `.csv`, `.dat`, or custom extensions without code changes

**Files:** `config/settings.py`, `mcp_server/ingestion.py`

### B3. Conditional code-specific analysis

**New setting:** `RAG_CONTENT_TYPE`: `"auto"` (default) | `"code"` | `"general"`

**Auto-detection logic:**
- Scan root directory for manifest files (pyproject.toml, package.json, go.mod, Cargo.toml, etc.)
- If any found → treat as code project → enable signatures, deps, manifests
- If none found → treat as general → skip code-specific tools

**When general:**
- `get_file_signatures` returns: "Signature extraction is only available for code projects. Set RAG_CONTENT_TYPE=code to force enable."
- `get_dependency_graph` returns similar message
- Overview skips manifest parsing, shows generic file stats instead

**Files:** `config/settings.py`, `mcp_server/analysis/overview.py`, `mcp_server/analysis/structure.py`, `mcp_server/tools/structure.py`, `mcp_server/tools/context.py`

### B4. Generalize overview analysis

**For all content types:**
- File count, total size, file type distribution (by extension)
- Directory tree (already generic)
- Top-level organization summary

**For code projects only (conditional):**
- Language breakdown with line counts
- Manifest detection and dependency listing
- Key files detection (entry points, CI, tests)

**Files:** `mcp_server/analysis/overview.py`

### B5. Rename MCP tools

| Current | New | Alias |
|---------|-----|-------|
| `search_codebase` | `search` | keep old name as alias |
| `search_codebase_by_file` | `search_by_file` | keep old name as alias |
| `get_codebase_context` | `get_context` | keep old name as alias |
| `get_file_signatures` | (unchanged) | n/a |
| `get_dependency_graph` | (unchanged) | n/a |
| `collection_stats` | (unchanged) | n/a |
| `ingest_current_directory` | `ingest` | keep old name as alias |
| `check_index_status` | (unchanged) | n/a |

Update all tool docstrings to be content-agnostic ("search indexed files" not "search codebase").

**Files:** `mcp_server/server.py`, `mcp_server/tools/*.py`

### B6. Add embedding batching

**Problem:** One embedding API call per chunk. Slow for large directories, expensive for cloud providers.

**Fix:**
- Add `embed_batch(texts: list[str]) -> list[list[float]]` to embedding base class
- Default implementation: loop over `embed()` (backward compatible)
- ONNX provider: batch inference natively
- OpenAI/Voyage: batch via API (they support multiple inputs per call)
- Ollama: batch if API supports it, else loop
- Update ingestion to batch chunks (32 at a time) before embedding
- Also add `get_embedding_batch(texts)` to `mcp_server/embeddings/__init__.py` module-level API (ingestion calls through this, not the provider directly)

**Files:** `mcp_server/embeddings/__init__.py`, `mcp_server/embeddings/base.py`, `mcp_server/embeddings/onnx_local.py`, `mcp_server/embeddings/openai_embed.py`, `mcp_server/embeddings/voyage.py`, `mcp_server/embeddings/ollama.py`, `mcp_server/ingestion.py`

### B7. Generalize git dependency

**Problem:** Non-git directories get no incremental updates, forcing full re-ingestion every time.

**Fix:**
- Check for `.git/` directory at ingestion time
- If git available: use current git-based change detection (unchanged)
- If no git: use file mtime + size comparison against a stored manifest (`.rag-mcp/file_manifest.json`)
- Manifest stores `{path: {mtime, size, hash}}` for each indexed file
- On re-ingestion, compare manifest to current state, only re-index changed/new files, delete removed files

**Files:** `mcp_server/ingestion.py`, new utility for manifest-based change detection

---

## Verification Plan

### Phase A verification:
1. Run full test suite: `python -m pytest tests/ -v --cov --cov-fail-under=65`
2. Test incremental ingestion: modify a file, re-ingest, verify old chunks are deleted and only new chunks exist
3. Test timeout behavior: set a very short timeout, verify partial state is recoverable
4. Test configurable limits: set custom `RAG_MAX_FILE_SIZE_BYTES`, `RAG_SKIP_DIRECTORIES`, verify they're respected
5. Verify no silent failures: check logs for proper warning messages on parse errors

### Phase B verification:
1. Run full test suite with updated names
2. Test on a non-code directory (e.g., a folder of markdown files): verify ingestion, search, and overview work
3. Test auto-detection: index a Python project (should enable code tools), then a docs folder (should disable them)
4. Test backward compatibility: verify old tool names still work as aliases
5. Test embedding batching: compare ingestion time before/after on a moderate-size directory
6. Test non-git directory: create a temp folder without git, verify incremental updates work via manifest
7. Test migration: create a `.codebase-rag/` directory, run the new tool, verify it's renamed to `.rag-mcp/`

---

## Key Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `config/settings.py` | A5, B1, B2, B3 | New settings, rename defaults |
| `mcp_server/ingestion.py` | A1, A3, A5, A6, B2, B6, B7 | Core changes to ingestion pipeline |
| `mcp_server/qdrant_client.py` | A1, A7 | File-level deletion, native filtering |
| `mcp_server/analysis/overview.py` | A2, A4, B3, B4 | Error handling, cache, conditional analysis |
| `mcp_server/analysis/structure.py` | A2, B3 | Error handling, conditional enable |
| `mcp_server/server.py` | B1, B5 | Rename, tool aliases |
| `mcp_server/cli.py` | B1 | Rename CLI commands |
| `mcp_server/embeddings/base.py` | B6 | Add `embed_batch()` interface |
| `mcp_server/embeddings/*.py` | B6 | Implement batching per provider |
| `mcp_server/tools/*.py` | B3, B5 | Conditional tools, renamed docstrings |
| `pyproject.toml` | B1 | Package name, CLI entry points |
| `tests/*.py` | All | Update for new behavior |

## Out of Scope

- Non-filesystem content sources (APIs, databases, URLs)
- PDF/image content extraction
- Semantic/AST-aware chunking (future enhancement)
- Multi-collection/multi-tenant support
- Search reranking or hybrid search (BM25 + vector)
- Query expansion or synonym support
