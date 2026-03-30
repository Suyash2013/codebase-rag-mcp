"""Unit tests for MCP tool functions in mcp_server/tools/."""

import json
from unittest.mock import patch

# ---------------------------------------------------------------------------
# search_codebase
# ---------------------------------------------------------------------------


def test_search_codebase_returns_formatted_results():
    """search_codebase should format results from search_chunks()."""
    mock_hits = [
        {"text": "def hello(): pass", "file_path": "main.py", "score": 0.95, "directory": "/proj"},
        {"text": "def world(): pass", "file_path": "utils.py", "score": 0.80, "directory": "/proj"},
    ]

    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch("mcp_server.tools.search.needs_ingestion", return_value=False),
        patch("mcp_server.tools.search.get_embedding", return_value=[0.1] * 384),
        patch("mcp_server.tools.search.search_chunks", return_value=mock_hits),
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.hybrid_search_enabled = False
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase

        result = search_codebase("hello function")

    assert "main.py" in result
    assert "utils.py" in result
    assert "0.950" in result


def test_search_codebase_no_results():
    """search_codebase should handle no results gracefully."""
    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch("mcp_server.tools.search.needs_ingestion", return_value=False),
        patch("mcp_server.tools.search.get_embedding", return_value=[0.1] * 384),
        patch("mcp_server.tools.search.search_chunks", return_value=[]),
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.hybrid_search_enabled = False
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase

        result = search_codebase("nonexistent thing")

    assert "No results found" in result


def test_search_codebase_triggers_ingestion_when_needed():
    """search_codebase should call ingest_directory when needs_ingestion returns True."""
    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch("mcp_server.tools.search.needs_ingestion", return_value=True),
        patch("mcp_server.tools.search.ingest_directory") as mock_ingest,
        patch("mcp_server.tools.search.get_embedding", return_value=[0.1] * 384),
        patch("mcp_server.tools.search.search_chunks", return_value=[]),
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.hybrid_search_enabled = False
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase

        search_codebase("test query")

    mock_ingest.assert_called_once_with("/proj")


def test_search_codebase_handles_exception():
    """search_codebase should return error string on exception."""
    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch(
            "mcp_server.tools.search.needs_ingestion", side_effect=RuntimeError("connection failed")
        ),
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase

        result = search_codebase("test")

    assert "No results found" in result or "connection failed" in result


# ---------------------------------------------------------------------------
# search_codebase_by_file
# ---------------------------------------------------------------------------


def test_search_codebase_by_file_passes_pattern():
    """search_codebase_by_file should pass file_pattern to search_chunks()."""
    mock_hits = [
        {"text": "class Model:", "file_path": "models/user.py", "score": 0.9, "directory": "/proj"},
    ]

    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch("mcp_server.tools.search.needs_ingestion", return_value=False),
        patch("mcp_server.tools.search.get_embedding", return_value=[0.1] * 384),
        patch("mcp_server.tools.search.search_chunks", return_value=mock_hits) as mock_search,
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.hybrid_search_enabled = False
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase_by_file

        result = search_codebase_by_file("user model", "models")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args
    # search_chunks(query_embedding, limit, directory_filter, file_pattern)
    assert call_kwargs[1]["file_pattern"] == "models" or (len(call_kwargs[0]) > 3 and call_kwargs[0][3] == "models")
    assert "models/user.py" in result


def test_search_codebase_by_file_no_matches():
    """search_codebase_by_file should report when no files match the pattern."""
    with (
        patch("mcp_server.tools.search.settings") as mock_settings,
        patch("mcp_server.tools.search.needs_ingestion", return_value=False),
        patch("mcp_server.tools.search.get_embedding", return_value=[0.1] * 384),
        patch("mcp_server.tools.search.search_chunks", return_value=[]),
    ):
        mock_settings.default_n_results = 10
        mock_settings.max_n_results = 20
        mock_settings.hybrid_search_enabled = False
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.search import search_codebase_by_file

        result = search_codebase_by_file("anything", "nonexistent_dir")

    assert "No results found" in result


# ---------------------------------------------------------------------------
# get_codebase_context
# ---------------------------------------------------------------------------


def test_get_codebase_context_uses_cache():
    """get_codebase_context should use cached overview when available."""
    cached = {
        "directory": "/proj",
        "total_files": 42,
        "languages": [{"extension": ".py", "files": 20, "lines": 5000}],
        "manifests": [],
        "key_files": {},
        "dependencies": {},
        "structure": ["src/", "  main.py"],
    }

    with (
        patch("mcp_server.tools.context.settings") as mock_settings,
        patch("mcp_server.tools.context.load_cached_overview", return_value=cached),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.context import get_codebase_context

        result = get_codebase_context()

    assert "Codebase Overview" in result
    assert "(cached)" in result
    assert "Total files: 42" in result


def test_get_codebase_context_generates_fresh():
    """get_codebase_context should generate fresh overview when no cache."""
    overview = {
        "directory": "/proj",
        "total_files": 10,
        "languages": [{"extension": ".py", "files": 5, "lines": 1000}],
        "manifests": [],
        "key_files": {},
        "dependencies": {},
        "structure": [],
    }

    with (
        patch("mcp_server.tools.context.settings") as mock_settings,
        patch("mcp_server.tools.context.load_cached_overview", return_value=None),
        patch("mcp_server.tools.context.generate_overview", return_value=overview),
        patch("mcp_server.tools.context.save_overview") as mock_save,
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.context import get_codebase_context

        result = get_codebase_context()

    assert "(fresh)" in result
    mock_save.assert_called_once()


def test_get_codebase_context_handles_error():
    """get_codebase_context should return error string on exception."""
    with (
        patch("mcp_server.tools.context.settings") as mock_settings,
        patch(
            "mcp_server.tools.context.load_cached_overview", side_effect=RuntimeError("disk full")
        ),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.context import get_codebase_context

        result = get_codebase_context()

    assert "Error generating codebase overview" in result


# ---------------------------------------------------------------------------
# get_file_signatures
# ---------------------------------------------------------------------------


def test_get_file_signatures_with_python_files(tmp_path):
    """get_file_signatures should extract signatures from real Python files."""
    (tmp_path / "module.py").write_text(
        "def greet(name: str) -> str:\n    return f'Hello {name}'\n\nclass Greeter:\n    pass\n"
    )

    with patch("mcp_server.tools.structure.settings") as mock_settings:
        mock_settings.get_working_directory.return_value = str(tmp_path)

        from mcp_server.tools.structure import get_file_signatures

        result = get_file_signatures(".py")

    assert "greet" in result
    assert "Greeter" in result


def test_get_file_signatures_no_matches(tmp_path):
    """get_file_signatures should report when no signatures found."""
    (tmp_path / "empty.txt").write_text("Just some text.\n")

    with patch("mcp_server.tools.structure.settings") as mock_settings:
        mock_settings.get_working_directory.return_value = str(tmp_path)

        from mcp_server.tools.structure import get_file_signatures

        result = get_file_signatures(".py")

    assert "No signatures found" in result


# ---------------------------------------------------------------------------
# get_dependency_graph
# ---------------------------------------------------------------------------


def test_get_dependency_graph_with_imports(tmp_path):
    """get_dependency_graph should show internal dependencies."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "__init__.py").write_text("")
    (tmp_path / "config" / "settings.py").write_text("VALUE = 1\n")
    (tmp_path / "main.py").write_text("from config.settings import VALUE\n")

    with patch("mcp_server.tools.structure.settings") as mock_settings:
        mock_settings.get_working_directory.return_value = str(tmp_path)

        from mcp_server.tools.structure import get_dependency_graph

        result = get_dependency_graph()

    assert "main.py" in result
    assert "config" in result


def test_get_dependency_graph_no_deps(tmp_path):
    """get_dependency_graph should report when no internal deps found."""
    (tmp_path / "standalone.py").write_text("import os\n")

    with patch("mcp_server.tools.structure.settings") as mock_settings:
        mock_settings.get_working_directory.return_value = str(tmp_path)

        from mcp_server.tools.structure import get_dependency_graph

        result = get_dependency_graph()

    assert "No internal dependencies found" in result


# ---------------------------------------------------------------------------
# collection_stats
# ---------------------------------------------------------------------------


def test_collection_stats_returns_json():
    """collection_stats should return valid JSON from get_stats."""
    mock_stats = {
        "collection_name": "test_codebase",
        "exists": True,
        "total_points": 150,
        "vectors_count": 150,
        "status": "green",
        "embedding_provider": "onnx",
        "qdrant_mode": "local",
    }

    with patch("mcp_server.tools.stats.get_stats", return_value=mock_stats):
        from mcp_server.tools.stats import collection_stats

        result = collection_stats()

    parsed = json.loads(result)
    assert parsed["total_points"] == 150
    assert parsed["exists"] is True


def test_collection_stats_handles_error():
    """collection_stats should return error string on exception."""
    with patch("mcp_server.tools.stats.get_stats", side_effect=RuntimeError("no connection")):
        from mcp_server.tools.stats import collection_stats

        result = collection_stats()

    assert "Error getting collection stats" in result


# ---------------------------------------------------------------------------
# check_index_status
# ---------------------------------------------------------------------------


def test_check_index_status_indexed_no_changes():
    """check_index_status should recommend search when indexed and clean."""
    with (
        patch("mcp_server.tools.ingest.settings") as mock_settings,
        patch("mcp_server.tools.ingest.needs_ingestion", return_value=False),
        patch(
            "mcp_server.tools.ingest.check_local_changes",
            return_value={"is_git_repo": True, "has_changes": False, "details": "clean"},
        ),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.ingest import check_index_status

        result = check_index_status()

    parsed = json.loads(result)
    assert parsed["is_indexed"] is True
    assert "semantic search" in parsed["recommendation"].lower()


def test_check_index_status_indexed_with_changes():
    """check_index_status should recommend re-indexing when there are changes."""
    with (
        patch("mcp_server.tools.ingest.settings") as mock_settings,
        patch("mcp_server.tools.ingest.needs_ingestion", return_value=False),
        patch(
            "mcp_server.tools.ingest.check_local_changes",
            return_value={"is_git_repo": True, "has_changes": True, "changed_files": 3},
        ),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.ingest import check_index_status

        result = check_index_status()

    parsed = json.loads(result)
    assert parsed["is_indexed"] is True
    assert "re-indexing" in parsed["recommendation"].lower()


def test_check_index_status_not_indexed():
    """check_index_status should indicate ingestion needed when not indexed."""
    with (
        patch("mcp_server.tools.ingest.settings") as mock_settings,
        patch("mcp_server.tools.ingest.needs_ingestion", return_value=True),
        patch(
            "mcp_server.tools.ingest.check_local_changes",
            return_value={"is_git_repo": True, "has_changes": False},
        ),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.ingest import check_index_status

        result = check_index_status()

    parsed = json.loads(result)
    assert parsed["is_indexed"] is False
    assert "ingestion needed" in parsed["recommendation"].lower()


def test_check_index_status_handles_error():
    """check_index_status should return error string on exception."""
    with (
        patch("mcp_server.tools.ingest.settings") as mock_settings,
        patch("mcp_server.tools.ingest.needs_ingestion", side_effect=RuntimeError("fail")),
    ):
        mock_settings.get_working_directory.return_value = "/proj"

        from mcp_server.tools.ingest import check_index_status

        result = check_index_status()

    assert "Error checking index status" in result
