"""Tests for codebase overview generation."""

from mcp_server.analysis.overview import generate_overview, load_cached_overview, save_overview


def test_generate_overview(tmp_codebase):
    """Should generate a valid overview for a sample codebase."""
    overview = generate_overview(str(tmp_codebase))

    assert overview["total_files"] > 0
    assert len(overview["languages"]) > 0
    assert isinstance(overview["structure"], list)


def test_overview_detects_languages(tmp_codebase):
    """Should detect Python and JavaScript files."""
    overview = generate_overview(str(tmp_codebase))
    extensions = [line["extension"] for line in overview["languages"]]
    assert ".py" in extensions


def test_overview_builds_tree(tmp_codebase):
    """Should include directory tree."""
    overview = generate_overview(str(tmp_codebase))
    assert len(overview["structure"]) > 0


def test_save_and_load_overview(tmp_codebase):
    """Should save and load cached overview."""
    overview = generate_overview(str(tmp_codebase))
    save_overview(str(tmp_codebase), overview)

    loaded = load_cached_overview(str(tmp_codebase))
    assert loaded is not None
    assert loaded["total_files"] == overview["total_files"]


def test_load_cached_overview_missing(tmp_path):
    """Should return None when no cache exists."""
    loaded = load_cached_overview(str(tmp_path))
    assert loaded is None


def test_overview_detects_pyproject(tmp_codebase):
    """Should detect pyproject.toml as a manifest."""
    (tmp_codebase / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "1.0"\ndependencies = ["requests"]\n'
    )
    overview = generate_overview(str(tmp_codebase))
    manifest_files = [m["file"] for m in overview.get("manifests", [])]
    assert "pyproject.toml" in manifest_files
