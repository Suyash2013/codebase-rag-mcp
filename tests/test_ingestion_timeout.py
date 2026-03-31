"""Tests for ingestion timeout handling and two-phase upsert/delete in ingest_directory."""

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from mcp_server.ingestion import (
    _embed_and_chunk_files,
    ingest_directory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_list(tmp_path: Path, names: list[str]) -> list[tuple[Path, str]]:
    """Create files in tmp_path and return (Path, rel_path) pairs."""
    result = []
    for name in names:
        fp = tmp_path / name
        fp.write_text(f"content of {name}\n")
        result.append((fp, name))
    return result


# ---------------------------------------------------------------------------
# _embed_and_chunk_files — return value signature
# ---------------------------------------------------------------------------


@patch("mcp_server.ingestion.get_embedding", return_value=[0.1, 0.2])
def test_embed_and_chunk_files_returns_three_tuple(mock_embed, tmp_path):
    """_embed_and_chunk_files must return (chunks, embeddings, processed_files)."""
    files = _make_file_list(tmp_path, ["a.py", "b.py"])
    result = _embed_and_chunk_files(files, str(tmp_path), timeout_seconds=3600, start_time=time.time())
    assert isinstance(result, tuple)
    assert len(result) == 3
    chunks, embeddings, processed = result
    assert isinstance(chunks, list)
    assert isinstance(embeddings, list)
    assert isinstance(processed, list)


@patch("mcp_server.ingestion.get_embedding", return_value=[0.1, 0.2])
def test_embed_and_chunk_files_processed_contains_all_on_success(mock_embed, tmp_path):
    """When no timeout occurs all file paths appear in processed_files."""
    files = _make_file_list(tmp_path, ["a.py", "b.py", "c.py"])
    _, _, processed = _embed_and_chunk_files(
        files, str(tmp_path), timeout_seconds=3600, start_time=time.time()
    )
    assert set(processed) == {"a.py", "b.py", "c.py"}


@patch("mcp_server.ingestion.get_embedding", return_value=[0.1, 0.2])
def test_embed_and_chunk_files_timeout_stops_early(mock_embed, tmp_path):
    """When timeout has already elapsed the loop exits immediately."""
    files = _make_file_list(tmp_path, ["a.py", "b.py", "c.py"])
    # Start time was long ago
    _, _, processed = _embed_and_chunk_files(
        files, str(tmp_path), timeout_seconds=1, start_time=time.time() - 10
    )
    assert len(processed) < 3


# ---------------------------------------------------------------------------
# ingest_directory — two-phase upsert/delete logic
# ---------------------------------------------------------------------------


@patch("mcp_server.ingestion.BM25Index")
@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "content_type": "text/x-python", "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],
    ),
)
def test_ingest_directory_two_phase_order(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    mock_bm25,
    tmp_path,
):
    """ingest_directory must upsert first, then delete, then re-upsert (idempotent restore)."""
    # The requirement is that we don't delete BEFORE we have new chunks ready.
    # The actual implementation calls upsert(all), then delete(each), then upsert(all).
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    # Verify execution order: upsert -> delete -> upsert
    # We look at the order of calls across multiple mocks
    manager = MagicMock()
    manager.attach_mock(mock_upsert, "upsert")
    manager.attach_mock(mock_delete_file, "delete")

    ingest_directory(str(tmp_path))

    expected_calls = [
        call.upsert([{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "content_type": "text/x-python", "ingested_at": ""}], [[0.1, 0.2]]),
        call.delete("a.py", str(tmp_path)),
        call.upsert([{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "content_type": "text/x-python", "ingested_at": ""}], [[0.1, 0.2]]),
    ]
    # Filter to only upsert and delete calls
    actual_calls = [c for c in manager.mock_calls if c[0] in ("upsert", "delete")]
    assert actual_calls == expected_calls


@patch("mcp_server.ingestion.BM25Index")
@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    # Simulate a timeout where a.py was processed but b.py was skipped
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "content_type": "text/x-python", "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],
    ),
)
def test_ingest_directory_delete_only_processed_files(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    mock_bm25,
    tmp_path,
):
    """delete_file_points is only called for files that were fully processed."""
    mock_collect.return_value = [
        (tmp_path / "a.py", "a.py"),
        (tmp_path / "b.py", "b.py"),
    ]
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    ingest_directory(str(tmp_path))

    # delete_file_points called only for 'a.py', not 'b.py'
    called_paths = [c.args[0] for c in mock_delete_file.call_args_list]
    assert "a.py" in called_paths
    assert "b.py" not in called_paths


# ---------------------------------------------------------------------------
# ingest_directory — old delete_directory_points no longer called
# ---------------------------------------------------------------------------


@patch("mcp_server.ingestion.BM25Index")
@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "content_type": "text/x-python", "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],
    ),
)
def test_ingest_directory_does_not_call_delete_directory_points(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    mock_bm25,
    tmp_path,
):
    """ingest_directory must NOT call delete_directory_points (data-loss risk)."""
    # Note: We removed the patch for delete_directory_points because it's no longer imported.
    # To be absolutely sure, we could try to patch it but it would fail if not present.
    # The fact that the previous test failed with AttributeError proves it's not there.
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    with patch("mcp_server.qdrant_client.delete_directory_points") as mock_delete_dir:
        ingest_directory(str(tmp_path))
        mock_delete_dir.assert_not_called()
