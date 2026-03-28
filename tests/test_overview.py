"""Tests for codebase overview generation."""

import json
from unittest.mock import patch

from mcp_server.analysis.overview import (
    _compute_fingerprint,
    generate_overview,
    load_cached_overview,
    save_overview,
)


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


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


def test_compute_fingerprint_uses_git_hash(tmp_path):
    """Should return the git HEAD commit hash for a git repository."""
    fake_hash = "abc123def456abc123def456abc123def456abc1"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = fake_hash + "\n"
        fp = _compute_fingerprint(str(tmp_path))
    assert fp == fake_hash


def test_compute_fingerprint_fallback_for_non_git(tmp_path):
    """Should fall back to file_count:total_size for non-git directories."""
    # Write two files of known sizes
    (tmp_path / "a.txt").write_text("hello")  # 5 bytes
    (tmp_path / "b.txt").write_text("world!")  # 6 bytes

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 128  # git not a repo
        mock_run.return_value.stdout = ""
        fp = _compute_fingerprint(str(tmp_path))

    # Should be "2:<5+6>" = "2:11"
    parts = fp.split(":")
    assert len(parts) == 2
    assert parts[0] == "2"
    assert int(parts[1]) == 11


def test_compute_fingerprint_fallback_when_git_unavailable(tmp_path):
    """Should fall back to file_count:total_size when git subprocess raises."""
    (tmp_path / "x.txt").write_bytes(b"x" * 20)

    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        fp = _compute_fingerprint(str(tmp_path))

    parts = fp.split(":")
    assert parts[0] == "1"
    assert int(parts[1]) == 20


def test_compute_fingerprint_empty_git_output_falls_back(tmp_path):
    """Should fall back when git rev-parse returns empty output."""
    (tmp_path / "f.txt").write_bytes(b"a" * 3)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "   \n"  # whitespace-only — treated as empty
        fp = _compute_fingerprint(str(tmp_path))

    # Should have fallen back to file-count fingerprint
    parts = fp.split(":")
    assert parts[0] == "1"


# ---------------------------------------------------------------------------
# Cache invalidation tests
# ---------------------------------------------------------------------------


def test_save_overview_stores_fingerprint(tmp_path):
    """save_overview should persist a fingerprint alongside the overview data."""
    overview = {"total_files": 5, "languages": []}
    fake_hash = "deadbeef" * 5
    with patch("mcp_server.analysis.overview._compute_fingerprint", return_value=fake_hash):
        save_overview(str(tmp_path), overview)

    cache_path = tmp_path / ".rag-mcp" / "overview.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload["fingerprint"] == fake_hash
    assert payload["overview"]["total_files"] == 5


def test_load_cached_overview_returns_data_on_match(tmp_path):
    """load_cached_overview should return the cached overview when fingerprints match."""
    overview = {"total_files": 7, "languages": []}
    fake_hash = "cafebabe" * 5
    with patch("mcp_server.analysis.overview._compute_fingerprint", return_value=fake_hash):
        save_overview(str(tmp_path), overview)
        loaded = load_cached_overview(str(tmp_path))

    assert loaded is not None
    assert loaded["total_files"] == 7


def test_load_cached_overview_returns_none_on_fingerprint_mismatch(tmp_path):
    """load_cached_overview should return None when the fingerprint has changed."""
    overview = {"total_files": 3, "languages": []}

    # Save with one fingerprint
    with patch("mcp_server.analysis.overview._compute_fingerprint", return_value="old-hash"):
        save_overview(str(tmp_path), overview)

    # Load with a different fingerprint (project has changed)
    with patch("mcp_server.analysis.overview._compute_fingerprint", return_value="new-hash"):
        loaded = load_cached_overview(str(tmp_path))

    assert loaded is None


def test_load_cached_overview_returns_none_for_legacy_format(tmp_path):
    """load_cached_overview should return None for old-style cache files without fingerprint."""
    cache_dir = tmp_path / ".rag-mcp"
    cache_dir.mkdir()
    # Write a legacy cache: raw overview dict with no wrapping
    legacy = {"total_files": 10, "languages": []}
    (cache_dir / "overview.json").write_text(json.dumps(legacy))

    loaded = load_cached_overview(str(tmp_path))
    assert loaded is None


def test_save_and_load_roundtrip_with_real_fingerprint(tmp_codebase):
    """End-to-end: save then load in the same directory should succeed."""
    overview = generate_overview(str(tmp_codebase))
    save_overview(str(tmp_codebase), overview)
    loaded = load_cached_overview(str(tmp_codebase))
    assert loaded is not None
    assert loaded["total_files"] == overview["total_files"]
