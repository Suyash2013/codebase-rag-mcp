"""Tests for ingestion timeout handling and two-phase upsert/delete in ingest_directory."""

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from mcp_server.ingestion import (
    _embed_and_chunk_files,
    _get_last_indexed_commit,
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
    # Set start_time far in the past so timeout is already exceeded
    _, _, processed = _embed_and_chunk_files(
        files, str(tmp_path), timeout_seconds=0, start_time=time.time() - 9999
    )
    assert processed == []


@patch("mcp_server.ingestion.get_embedding", return_value=[0.1, 0.2])
def test_embed_and_chunk_files_timeout_logs_skipped(mock_embed, tmp_path, caplog):
    """On timeout a warning is emitted listing the skipped files."""
    import logging

    files = _make_file_list(tmp_path, ["a.py", "b.py"])
    with caplog.at_level(logging.WARNING, logger="rag-mcp"):
        _embed_and_chunk_files(
            files, str(tmp_path), timeout_seconds=0, start_time=time.time() - 9999
        )
    assert any("Skipped files" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ingest_directory — two-phase upsert/delete
# ---------------------------------------------------------------------------


@patch("mcp_server.ingestion.upsert_chunks", return_value=2)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [
            {"id": "uuid-1", "text": "x", "file_path": "a.py", "directory": "/tmp/d", "chunk_index": 0, "ingested_at": ""},
        ],
        [[0.1, 0.2]],
        ["a.py"],
    ),
)
def test_ingest_directory_upsert_called_before_delete(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    tmp_path,
):
    """ingest_directory should upsert new chunks BEFORE deleting old file points."""
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    call_order = []
    mock_upsert.side_effect = lambda *a, **kw: call_order.append("upsert") or 2
    mock_delete_file.side_effect = lambda *a, **kw: call_order.append("delete")

    ingest_directory(str(tmp_path))

    assert call_order[0] == "upsert", "First call must be upsert, not delete"
    assert "delete" in call_order, "delete_file_points must be called"


@patch("mcp_server.ingestion.upsert_chunks", return_value=2)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [
            {"id": "uuid-1", "text": "x", "file_path": "a.py", "directory": "/tmp/d", "chunk_index": 0, "ingested_at": ""},
        ],
        [[0.1, 0.2]],
        ["a.py"],
    ),
)
def test_ingest_directory_re_upserts_after_delete(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    tmp_path,
):
    """ingest_directory must re-upsert chunks after deleting stale points."""
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    ingest_directory(str(tmp_path))

    # upsert_chunks should be called twice (initial + re-upsert after delete)
    assert mock_upsert.call_count == 2


@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],  # 1 of 2 files processed (simulates timeout)
    ),
)
def test_ingest_directory_no_commit_marker_on_partial(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    tmp_path,
):
    """Commit marker must NOT be written when ingestion timed out (partial)."""
    # Report 2 files in _collect but _embed only processed 1 (partial)
    mock_collect.return_value = [
        (tmp_path / "a.py", "a.py"),
        (tmp_path / "b.py", "b.py"),
    ]
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")

    ingest_directory(str(tmp_path))

    # last_commit.txt must NOT have been created
    marker = tmp_path / ".rag-mcp" / "last_commit.txt"
    assert not marker.exists(), "Commit marker must not be written for partial ingestion"


@patch("mcp_server.ingestion._save_indexed_commit")
@patch("mcp_server.ingestion._get_current_commit", return_value="abc123")
@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],  # all 1 file processed successfully
    ),
)
def test_ingest_directory_writes_commit_marker_on_full_success(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
    mock_current_commit,
    mock_save_commit,
    tmp_path,
):
    """Commit marker IS written when all files are processed without timeout."""
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    ingest_directory(str(tmp_path))

    mock_save_commit.assert_called_once_with(str(tmp_path.resolve()), "abc123")


@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "ingested_at": ""}],
        [[0.1, 0.2]],
        ["a.py"],  # only a.py processed; b.py was skipped
    ),
)
def test_ingest_directory_only_deletes_processed_files(
    mock_embed,
    mock_collect,
    mock_dim,
    mock_ensure,
    mock_delete_file,
    mock_upsert,
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


@patch("mcp_server.ingestion.upsert_chunks", return_value=1)
@patch("mcp_server.ingestion.delete_directory_points")
@patch("mcp_server.ingestion.delete_file_points")
@patch("mcp_server.ingestion.ensure_collection")
@patch("mcp_server.ingestion.get_embedding_dimension", return_value=2)
@patch("mcp_server.ingestion._collect_text_files")
@patch(
    "mcp_server.ingestion._embed_and_chunk_files",
    return_value=(
        [{"id": "u1", "text": "x", "file_path": "a.py", "directory": "", "chunk_index": 0, "ingested_at": ""}],
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
    mock_delete_dir,
    mock_upsert,
    tmp_path,
):
    """ingest_directory must NOT call delete_directory_points (data-loss risk)."""
    mock_collect.return_value = [(tmp_path / "a.py", "a.py")]
    (tmp_path / "a.py").write_text("x = 1\n")

    ingest_directory(str(tmp_path))

    mock_delete_dir.assert_not_called()
