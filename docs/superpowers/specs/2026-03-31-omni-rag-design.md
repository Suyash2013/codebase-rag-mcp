# Omni-RAG: Transform codebase-rag-mcp into a General-Purpose RAG MCP Plugin

**Date:** 2026-03-31
**Status:** Approved
**Supersedes:** 2026-03-29-general-purpose-rag-mcp-design.md

## Context

`codebase-rag-mcp` is a solid semantic search MCP plugin, but it's code-only. Our rival GitNexus does code analysis better (knowledge graphs, impact analysis, execution flows), but is also code-only. Our differentiator: **general-purpose "index anything"** — PDFs, Word docs, CSVs, images, markdown, code, configs, logs, research papers.

The `feature/general-purpose-rag` branch already has Phase A bug fixes and a partial rename to `rag-mcp`. This spec builds on that work, renaming to `omni-rag` and adding major new capabilities.

## Design Decisions

| Decision | Choice |
|---|---|
| Target content | All file types ("index anything") |
| Structural awareness | Hybrid — format-aware chunking per content type |
| Non-git support | Git-aware when available, file-hash fallback |
| Extractor architecture | Plugin system with clean interface, optional extras |
| Search | BM25 + semantic with reciprocal rank fusion |
| Tool split | Core (6 tools) + code plugin (2 tools, enabled by default) |
| Rename | `omni-rag-mcp`, `OMNI_RAG_` prefix, `omni-rag` CLI |
| Language | Stay with Python |

---

## Architecture

### Module Structure

```
omni_rag/                          # renamed from mcp_server/
    __init__.py
    server.py                      # Core MCP server + conditional code plugin registration
    cli.py                         # CLI entry points (omni-rag, omni-rag-setup)
    ingestion.py                   # Orchestrator (~100 lines, delegates to extractors/chunkers)
    storage/
        __init__.py
        qdrant_client.py           # Vector store operations (moved from mcp_server/)
        bm25_index.py              # BM25 sparse index (NEW)
        hybrid.py                  # Reciprocal rank fusion (NEW)
    embeddings/                    # Unchanged structure
        __init__.py
        base.py
        factory.py
        onnx_local.py
        ollama.py
        openai.py
        voyage.py
    extractors/                    # NEW: content extraction plugin system
        __init__.py
        base.py                    # ExtractorBase ABC + registry
        text.py                    # Plain text fallback (.txt, .log, etc.)
        code.py                    # Source code (current 79 extensions)
        markdown.py                # .md, .rst, .adoc
        structured_data.py         # JSON, YAML, CSV, TOML, XML
        # --- optional extras (pip install omni-rag-mcp[pdf]) ---
        pdf.py                     # Requires pymupdf
        docx.py                    # Requires python-docx
        image.py                   # Requires pytesseract + Pillow
    chunkers/                      # NEW: format-aware chunking
        __init__.py
        base.py                    # ChunkerBase ABC + registry
        recursive.py               # Current recursive char splitter (fallback)
        code_chunker.py            # AST-boundary-aware splitting
        markdown_chunker.py        # Heading-aware splitting
        structured_chunker.py      # JSON keys / CSV row-groups / YAML top-level
        paragraph_chunker.py       # Paragraph/section boundary splitting (PDF, DOCX)
    change_detection/              # NEW: abstracted change detection
        __init__.py
        base.py                    # ChangeDetector ABC + factory
        git_detector.py            # Current git logic (refactored out of ingestion.py)
        file_hash_detector.py      # Mtime + hash fallback for non-git dirs
    analysis/
        __init__.py
        overview.py                # Generalized project overview
        structure.py               # Code-only: AST/regex signature extraction (unchanged)
    tools/                         # Preserve existing per-concern file structure
        __init__.py
        search.py                  # search, search_by_file (renamed from search_codebase)
        context.py                 # get_context (renamed from get_codebase_context)
        ingest.py                  # ingest (renamed from ingest_current_directory)
        stats.py                   # stats, check_status (renamed)
        structure.py               # get_file_signatures, get_dependency_graph (conditionally registered)
config/
    __init__.py
    settings.py                    # Updated: OMNI_RAG_ prefix, new settings, migration shim
```

### Key Interfaces

**ExtractorBase** (`extractors/base.py`):
- `supported_extensions() -> set[str]`
- `supported_filenames() -> set[str]`
- `can_extract(path: Path) -> bool`
- `extract(path: Path) -> ExtractionResult`
- `max_file_size() -> int` (default 1MB)
- `ExtractionResult`: `text`, `content_type` ("code"|"markdown"|"structured_data"|"document"|"plain_text"), `metadata`
- Global registry: `register_extractor()`, `get_extractor(path)` — checks filename, then extension, then fallback

**ChunkerBase** (`chunkers/base.py`):
- `content_types() -> set[str]`
- `chunk(text, chunk_size, chunk_overlap, metadata?) -> list[Chunk]`
- `Chunk`: `text`, `metadata` (section_title, line_range, etc.)
- Registry: `register_chunker()`, `get_chunker(content_type)` — falls back to recursive chunker

**Extractor -> Chunker routing table**:
| Extractor | content_type | Chunker | Notes |
|---|---|---|---|
| `CodeExtractor` | `"code"` | `CodeChunker` | AST splitting for Python; regex-boundary for JS/TS/Go/Java/Kotlin; falls back to `RecursiveChunker` for other languages |
| `MarkdownExtractor` | `"markdown"` | `MarkdownChunker` | Splits on heading boundaries, sub-splits oversized sections by paragraph |
| `StructuredDataExtractor` | `"structured_data"` | `StructuredChunker` | JSON/YAML: top-level keys. CSV: row groups with header. TOML/XML: sections |
| `PdfExtractor` | `"document"` | `ParagraphChunker` | Paragraph/section boundary splitting |
| `DocxExtractor` | `"document"` | `ParagraphChunker` | Same as PDF |
| `ImageExtractor` | `"plain_text"` | `RecursiveChunker` | OCR output is plain text |
| `TextExtractor` | `"plain_text"` | `RecursiveChunker` | Default fallback |

**ChangeDetector** (`change_detection/base.py`):
- `detect_changes(directory) -> ChangeReport`
- `save_checkpoint(directory)`
- `has_checkpoint(directory) -> bool`
- `ChangeReport`: `has_changes`, `changed_files`, `deleted_files`, `details`
- Factory: `create_detector(directory)` — returns GitDetector if `.git/` exists, else FileHashDetector

**Code plugin registration** (no PluginBase ABC — YAGNI until a second plugin exists):
- `server.py` conditionally imports and registers tools from `tools/structure.py` based on `settings.enable_code_plugin`
- Simple `if settings.enable_code_plugin:` guard, no abstraction layer

### Hybrid Search

- **BM25 implementation**: Custom lightweight BM25-Okapi implementation (avoid `rank_bm25` which pulls in NumPy). ~100 lines of pure Python, no extra dependencies. Serialized to `.omni-rag/bm25_index.json`. Lazy-loaded on first search.
- **Search flow**: Query -> parallel (Qdrant semantic search + BM25 keyword search) -> Reciprocal Rank Fusion -> merged results
- **RRF formula**: `score = weight / (k + rank)` summed across both lists
- **Default weights**: 70% semantic, 30% BM25. Configurable via `OMNI_RAG_HYBRID_SEMANTIC_WEIGHT` / `OMNI_RAG_HYBRID_BM25_WEIGHT`
- **BM25 corpus persistence**: During ingestion, chunk texts and their IDs are stored in `.omni-rag/bm25_corpus.json` alongside the BM25 index. On incremental re-index: load corpus, remove deleted chunk IDs, add new chunks, rebuild BM25 index.
- **Incremental updates**: Load persisted corpus -> apply deltas (add/remove changed files' chunks) -> rebuild BM25 index -> save both corpus and index

### Qdrant Payload Change

Current: `{text, file_path, directory, chunk_index, ingested_at}`
New (additive): `{..., content_type, section_title, extractor}` — enables filtered search by content type

### Core vs Code Plugin Tools

**Core (always)**:
| Current | New | Alias (deprecated) |
|---|---|---|
| `search_codebase` | `search` | `search_codebase` registered as alias for one release |
| `search_codebase_by_file` | `search_by_file` | `search_codebase_by_file` alias |
| `get_codebase_context` | `get_context` | `get_codebase_context` alias |
| `collection_stats` | `stats` | `collection_stats` alias |
| `ingest_current_directory` | `ingest` | `ingest_current_directory` alias |
| `check_index_status` | `check_status` | `check_index_status` alias |

Old tool names are registered as aliases (thin wrappers that call the new function + log a deprecation warning). Removed after one release cycle.

**Code plugin (default enabled, `OMNI_RAG_ENABLE_CODE_PLUGIN=false` to disable)**:
- `get_file_signatures`
- `get_dependency_graph`

### Config Changes

- Prefix: `RAG_` -> `OMNI_RAG_` (dual support for one release)
- Local path: `.codebase-rag/` -> `.omni-rag/`
- New settings: `hybrid_search_enabled` (true), `hybrid_semantic_weight` (0.7), `hybrid_bm25_weight` (0.3), `enable_code_plugin` (true)
- Optional dependency settings auto-detected (no config needed)

### Migration

- `.codebase-rag/` (and `.rag-mcp/` from prior branch work) auto-renamed to `.omni-rag/` on first startup
- `RAG_` env vars mapped to `OMNI_RAG_` with deprecation warning
- Old CLI commands kept as aliases for one release
- Default collection name -> `"omni_rag"`. On startup: if `"omni_rag"` collection doesn't exist but `"codebase"` or `"documents"` does, use it automatically (no re-ingest). Qdrant doesn't support collection rename, so we just detect and use the old name.

### Zero-Config Guarantee

Default experience unchanged:
- `pip install omni-rag-mcp` -> ONNX embeddings, local Qdrant, BM25, all built-in extractors
- `omni-rag` -> just works, auto-ingests on first search
- No external services required
- Optional extras only for PDF/DOCX/image support

---

## Implementation Phases

### Phase 1: Refactor (no rename yet, tests keep passing)
1. Extract `RecursiveChunker` from `ingestion.py` into `chunkers/recursive.py`
2. Extract git logic from `ingestion.py` into `change_detection/git_detector.py`
3. Create `ExtractorBase` ABC + `TextExtractor` + `CodeExtractor` (wrapping current logic)
4. Refactor `ingestion.py` to use extractor/chunker/detector abstractions
5. Implement `FileHashDetector` for non-git directories

### Phase 2: New capabilities
6. Add format-aware chunkers: `CodeChunker`, `MarkdownChunker`, `StructuredChunker`, `ParagraphChunker`
7. Add `MarkdownExtractor` and `StructuredDataExtractor`
8. Implement `BM25Index` + `hybrid.py` rank fusion
9. Wire hybrid search into search tools
10. Add optional extractors (PDF, DOCX, image) with graceful dependency handling

### Phase 3: Rename + conditional code tools
11. Rename `mcp_server/` -> `omni_rag/`, update all imports
12. Rename tools (with deprecated aliases), CLI commands, config prefix
13. Add `enable_code_plugin` config guard around `tools/structure.py` registration in `server.py`
14. Add migration helpers (env var compat, directory rename, Qdrant collection auto-detect)
15. Qdrant collection migration: on startup, check if `"omni_rag"` collection exists; if not, check for `"codebase"` or `"documents"` collection and use it (no re-ingest needed)

### Phase 4: Polish
16. Update all tests (rename fixtures, add tests for new components)
17. Update CLAUDE.md, pyproject.toml, CLI help
18. Add deprecation warnings for old env vars/commands
19. Full test suite pass

---

## Verification

1. All existing tests pass after each phase
2. `omni-rag` CLI starts and serves MCP tools
3. Semantic search works on mixed content (code + markdown + text)
4. BM25 + semantic hybrid returns better results than semantic-only for keyword queries
5. Non-git directory indexing works via file hash detection
6. Optional extras (PDF etc.) gracefully skip when not installed
7. Migration from `.codebase-rag/` / `.rag-mcp/` to `.omni-rag/` is automatic
8. Old `RAG_` env vars still work with deprecation warning
9. Full test suite passes: `python -m pytest tests/ -v`
