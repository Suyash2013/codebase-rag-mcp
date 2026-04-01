# Project Cleanup & PyPI Release Mechanism

**Date:** 2026-04-01
**Status:** Draft

## Context

omni-rag-mcp (v3.0.0) is ready for its first PyPI release. The project has accumulated development artifacts (empty modules, AI agent metadata, stale design docs, deprecated CLI aliases) that should be cleaned before publishing. The CI pipeline already has a publish job using OIDC trusted publishing on `v*` tags, but there's no release script or GitHub Release automation.

Multiple files still reference the old project names (`codebase-rag-mcp`, `rag-mcp`) and old env var prefix (`RAG_`).

## Part 1: Project Cleanup

### 1A. Files & Directories to Delete

| Item | Reason |
|------|--------|
| `utils/` (entire directory) | Empty — only contains `__init__.py`, zero imports anywhere |
| `.agent/` (entire directory, ~40 files) | AI tooling metadata, not project code |
| `docs/plans/` (entire directory) | Stale implementation trackers — `task.md` shows items as "not_started" that are done; `2026-03-29-general-purpose-rag.md` is superseded |
| `docs/superpowers/specs/2026-03-29-general-purpose-rag-mcp-design.md` | Superseded by `2026-03-31-omni-rag-design.md` |
| `docs/superpowers/specs/2026-03-31-omni-rag-design.md` | Historical design artifact, not user-facing |
| `docs/superpowers/plans/2026-03-31-omni-rag-implementation.md` | Historical implementation plan, not user-facing |

After cleanup, `docs/superpowers/` will only contain this spec and future specs (in git only — excluded from sdist/wheel).

### 1B. .gitignore Additions

```
.agent/
.omni-rag/
```

### 1C. Remove Deprecated Console Scripts

In `pyproject.toml`, remove:
```toml
# Deprecated aliases
rag-mcp = "mcp_server.cli:main"
rag-mcp-setup = "mcp_server.cli:setup"
```

Also in `mcp_server/cli.py`: remove the legacy fallback `shutil.which("rag-mcp")` on line 25 (the primary lookup on line 22 already uses `omni-rag`).

### 1D. Fix All Stale References

**README.md** — Full rewrite needed. Current state is entirely stale:
- Line 1: Title `codebase-rag-mcp` → `omni-rag-mcp`
- Line 3: Description still says "codebase search" → update to "general-purpose RAG"
- Line 8: `pip install codebase-rag-mcp` → `pip install omni-rag-mcp`
- Line 9: `codebase-rag-setup` → `omni-rag-setup`
- Lines 29-37: Tool names table uses deprecated names (`search_codebase`, `get_codebase_context`, `collection_stats`, `ingest_current_directory`, `check_index_status`) → update to current names (`search`, `search_by_file`, `get_context`, `get_file_signatures`, `get_dependency_graph`, `stats`, `ingest`, `check_status`) matching CLAUDE.md
- Lines 46-48: Env var prefix `RAG_` → `OMNI_RAG_`
- Line 52: Storage path `.codebase-rag/` → `.omni-rag/`
- Lines 56-58: Env vars `RAG_QDRANT_MODE` etc. → `OMNI_RAG_QDRANT_MODE` etc.
- Line 63: `RAG_` prefix → `OMNI_RAG_` prefix
- Line 80: `codebase-rag-setup` → `omni-rag-setup`
- Lines 83-88: MCP config JSON `codebase-rag` → `omni-rag`

**CONTRIBUTING.md** — Multiple fixes:
- Line 1: `Contributing to codebase-rag-mcp` → `Contributing to omni-rag-mcp`
- Line 8: `cd codebase-rag-mcp` → `cd omni-rag-mcp`
- Line 43: `with the RAG_ prefix` → `with the OMNI_RAG_ prefix`
- Line 46: `RAG_* variables` → `OMNI_RAG_* variables`

**config/.env.example** — Entire file uses old prefix:
- Line 1: `# codebase-rag-mcp configuration` → `# omni-rag-mcp configuration`
- Line 2: `All settings use the RAG_ prefix` → `All settings use the OMNI_RAG_ prefix`
- All variable names: `RAG_*` → `OMNI_RAG_*` (lines 6-41)
- Line 7: `.codebase-rag/qdrant` → `.omni-rag/qdrant`
- Line 17: `.codebase-rag/models/` → `.omni-rag/models/`

**config/mcp_config.example.json** — Update server key and command:
- `"codebase-rag"` → `"omni-rag"` (both key and command value)

**mcp_server/extractors/pdf.py** — Docstring on line 17:
- `pip install rag-mcp[pdf]` → `pip install omni-rag-mcp[pdf]`

**scripts/health_check.py** — Two fixes:
- Line 89: `Set RAG_{provider.upper()}_API_KEY` → `Set OMNI_RAG_{provider.upper()}_API_KEY`
- Line 96: `Codebase RAG MCP` → `omni-rag-mcp`

**config/.env.example** — Add missing documented settings (present in `config/settings.py` and CLAUDE.md but absent from example):
```
# OMNI_RAG_HYBRID_SEARCH_ENABLED=true   # Enable BM25+semantic hybrid search
# OMNI_RAG_ENABLE_CODE_PLUGIN=true      # Enable code-specific tools (signatures, dependency graph)
```

### 1E. Hatch Build Exclusions

Add to `pyproject.toml` to ensure non-source files don't end up in the sdist:
```toml
[tool.hatch.build.targets.sdist]
exclude = [
    ".agent/",
    ".github/",
    ".idea/",
    ".claude/",
    ".worktrees/",
    ".rag-mcp/",
    "docs/plans/",
    "docs/superpowers/",
]
```

## Part 2: Release Script

**File:** `scripts/release.py`

### Behavior

```
python scripts/release.py 3.1.0          # bump, commit, tag, push
python scripts/release.py --dry-run 3.1.0  # show what would happen, no changes
```

1. **Parse args** — version (required), `--dry-run` flag (optional)
2. **Validate** version matches `X.Y.Z` or `X.Y.ZrcN` pattern (PEP 440 compliant)
3. **Read** `pyproject.toml`, extract current version, confirm it's different
4. **Check** working tree is clean (no uncommitted changes besides pyproject.toml)
5. **Check** tag `v{version}` doesn't already exist
6. **Update** the `version = "..."` line in `pyproject.toml`
7. **Commit** `git add pyproject.toml && git commit -m "release: v{version}"`
8. **Tag** `git tag -a v{version} -m "Release v{version}"`
9. **Push** `git push origin HEAD && git push origin v{version}` (uses HEAD, not hardcoded branch)
10. Print summary: "Tagged v{version} — CI will build and publish to PyPI"

In `--dry-run` mode, print each step but skip all git mutations (steps 6-9).

### Error Handling

- No version argument → print usage and exit
- Invalid version format → error message and exit
- Dirty working tree → warn and abort
- Tag already exists → error and abort
- Git push fails → error (tag is local, user can retry)

### Dependencies

None beyond Python stdlib — uses `subprocess`, `re`, `sys`, `pathlib`.

## Part 3: GitHub Release Automation

**File:** `.github/workflows/release.yml`

Triggers after CI completes on `v*` tags using `workflow_run` to avoid race conditions (ensures package is published to PyPI before the GitHub Release is created).

```yaml
name: Release

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    # branches-ignore: ["**"] ensures only tag-triggered CI runs activate this workflow.
    # Tags are not branches, so they bypass branch filters entirely.
    branches-ignore: ["**"]

permissions:
  contents: write

jobs:
  github-release:
    name: Create GitHub Release
    if: >-
      github.event.workflow_run.conclusion == 'success' &&
      startsWith(github.event.workflow_run.head_branch, 'v')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "${{ github.event.workflow_run.head_branch }}" \
            --generate-notes \
            --title "${{ github.event.workflow_run.head_branch }}"
```

## Part 4: PyPI OIDC Setup (Manual)

The user must configure trusted publishing on PyPI:

1. Go to https://pypi.org → `omni-rag-mcp` project → Settings → Publishing
2. Add trusted publisher:
   - **Owner:** `Suyash2013`
   - **Repository:** `codebase-rag-mcp` (GitHub repo name, not PyPI package name)
   - **Workflow:** `ci.yml`
   - **Environment:** `pypi`

The CI already has the correct publish job configuration with `environment: pypi` and `id-token: write` permissions.

## Release Flow (End-to-End)

```
Developer                    CI (ci.yml)                PyPI
   |                         |                           |
   |-- scripts/release.py -->|                           |
   |   (bump, commit, tag,   |                           |
   |    push)                |                           |
   |                         |-- lint ------------------>|
   |                         |-- typecheck ------------->|
   |                         |-- security audit -------->|
   |                         |-- test (3.10/3.11/3.12) ->|
   |                         |-- hatch build ----------->|
   |                         |-- pypa/gh-action-pypi --->|-- publish
   |                         |                           |
   |                    CI completes                     |
   |                         |                           |
   |                    release.yml triggers             |
   |                         |-- gh release create ----->|
   |                         |   (auto-generated notes)  |
```

## Implementation Order

1. **Part 1 first** — cleanup and doc fixes (single commit)
2. **Parts 2 & 3** — release script + workflow (single commit)
3. **Part 4** — manual PyPI OIDC setup by user
4. **Test** — push an RC tag (e.g., `v3.0.1rc1`) to verify the full pipeline

Parts 2 & 3 modify `pyproject.toml` (Part 1 also modifies it), so Part 1 should land first to avoid conflicts.

## Verification

1. **Cleanup:** `git status` shows deleted files; `hatch build` produces clean sdist/wheel; inspect contents with `tar -tzf dist/*.tar.gz` to confirm no deleted artifacts
2. **Release script:** `python scripts/release.py --dry-run 3.0.1` shows correct steps without executing
3. **CI pipeline:** Push tag `v3.0.1rc1` → verify lint/test/build/publish jobs run
4. **GitHub Release:** After CI completes, verify release appears at `github.com/Suyash2013/omni-rag-mcp/releases`

**Note on bad releases:** PyPI does not allow re-uploading a deleted version. If a bad version is published, the fix is to publish a new patch version (e.g., 3.0.2 to fix 3.0.1).

## Files to Modify/Create

| Action | File |
|--------|------|
| Delete | `utils/`, `.agent/`, `docs/plans/`, `docs/superpowers/specs/2026-03-29-*.md`, `docs/superpowers/specs/2026-03-31-*.md`, `docs/superpowers/plans/` |
| Edit | `pyproject.toml` — remove deprecated scripts, add sdist exclusions |
| Edit | `.gitignore` — add `.agent/`, `.omni-rag/` |
| Edit | `README.md` — full rewrite with current project name, tools, env vars, storage path |
| Edit | `CONTRIBUTING.md` — update project name and env var prefix references |
| Edit | `config/.env.example` — update all `RAG_*` to `OMNI_RAG_*`, fix paths |
| Edit | `config/mcp_config.example.json` — update server name and command |
| Edit | `mcp_server/cli.py` — remove legacy `rag-mcp` fallback (line 25) |
| Edit | `mcp_server/extractors/pdf.py` — fix install instruction in docstring |
| Edit | `scripts/health_check.py` — fix env var prefix and title |
| Create | `scripts/release.py` — version bump + tag + push script with `--dry-run` |
| Create | `.github/workflows/release.yml` — GitHub Release automation (workflow_run trigger) |
