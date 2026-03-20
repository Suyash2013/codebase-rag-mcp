"""MCP tool for codebase context and overview."""

from config.settings import settings
from mcp_server.analysis.overview import generate_overview, load_cached_overview, save_overview


def get_codebase_context() -> str:
    """Get a compressed overview of the codebase structure.

    Returns language breakdown, directory structure, key files, and
    dependency information. Use this FIRST when starting work on an
    unfamiliar codebase to understand its layout before searching
    for specific code.

    Much cheaper than reading files individually — gives you the big
    picture in one call.
    """
    directory = settings.get_working_directory()

    try:
        # Try cached first
        overview = load_cached_overview(directory)
        if overview:
            return _format_overview(overview, cached=True)

        # Generate fresh
        overview = generate_overview(directory)
        save_overview(directory, overview)
        return _format_overview(overview, cached=False)

    except Exception as exc:
        return f"Error generating codebase overview: {exc}"


def _format_overview(overview: dict, cached: bool) -> str:
    """Format overview into a readable string."""
    parts = []

    parts.append(f"# Codebase Overview {'(cached)' if cached else '(fresh)'}")
    parts.append(f"Directory: {overview['directory']}")
    parts.append(f"Total files: {overview['total_files']}")

    # Languages
    if overview.get("languages"):
        parts.append("\n## Languages (by file count)")
        for lang in overview["languages"][:10]:
            parts.append(
                f"  {lang['extension']:12s} {lang['files']:4d} files  {lang['lines']:6d} lines"
            )

    # Manifests
    if overview.get("manifests"):
        parts.append("\n## Project Type")
        for m in overview["manifests"]:
            parts.append(f"  {m['file']} → {m['language']}")

    # Key files
    if overview.get("key_files"):
        parts.append("\n## Key Files")
        for category, files in overview["key_files"].items():
            parts.append(f"  {category}: {', '.join(files[:5])}")

    # Dependencies
    if overview.get("dependencies"):
        parts.append("\n## Dependencies")
        for ecosystem, info in overview["dependencies"].items():
            name = info.get("name", info.get("module", ""))
            version = info.get("version", "")
            deps = info.get("dependencies", info.get("requires", []))
            parts.append(f"  {ecosystem}: {name} {version}")
            if deps:
                for d in deps[:15]:
                    parts.append(f"    - {d}")
                if len(deps) > 15:
                    parts.append(f"    ... and {len(deps) - 15} more")

    # Structure
    if overview.get("structure"):
        parts.append("\n## Directory Structure")
        for line in overview["structure"][:50]:
            parts.append(f"  {line}")
        if len(overview["structure"]) > 50:
            parts.append(f"  ... ({len(overview['structure']) - 50} more entries)")

    return "\n".join(parts)
