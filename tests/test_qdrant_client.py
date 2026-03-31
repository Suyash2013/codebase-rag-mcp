"""Tests for Qdrant client operations using in-memory Qdrant."""

import uuid
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from config.settings import settings


@pytest.fixture
def memory_client():
    """Create an in-memory Qdrant client for testing."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    return client


@pytest.fixture
def populated_client(memory_client):
    """Qdrant client with sample data."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={
                "text": "def hello(): print('hello')",
                "file_path": "main.py",
                "directory": "/test/project",
                "chunk_index": 0,
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        ),
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.5, 0.6, 0.7, 0.8],
            payload={
                "text": "const app = express()",
                "file_path": "app.js",
                "directory": "/test/project",
                "chunk_index": 0,
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        ),
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.9, 0.1, 0.2, 0.3],
            payload={
                "text": "SELECT * FROM users",
                "file_path": "queries.sql",
                "directory": "/other/project",
                "chunk_index": 0,
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        ),
    ]
    memory_client.upsert(collection_name=settings.qdrant_collection, points=points)
    return memory_client


def test_collection_creation(memory_client):
    """Collection should exist after creation."""
    collections = [c.name for c in memory_client.get_collections().collections]
    assert settings.qdrant_collection in collections


def test_search_chunks_returns_results(populated_client):
    """search_chunks should return matching results with scores."""
    from mcp_server.qdrant_client import ensure_collection, search_chunks
    with patch("mcp_server.qdrant_client.get_client", return_value=populated_client):
        ensure_collection(4)
        hits = search_chunks(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            limit=2,
        )
    assert len(hits) == 2
    assert hits[0]["file_path"] == "main.py"
    assert "score" in hits[0]


def test_search_chunks_with_filter(populated_client):
    """search_chunks with directory filter should restrict results."""
    from mcp_server.qdrant_client import ensure_collection, search_chunks
    with patch("mcp_server.qdrant_client.get_client", return_value=populated_client):
        ensure_collection(4)
        hits = search_chunks(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            limit=10,
            directory_filter="/test/project"
        )
    assert len(hits) == 2
    for h in hits:
        assert h["directory"] == "/test/project"


def test_search_chunks_with_file_pattern_filter(populated_client):
    """search_chunks with file_pattern filter should restrict to matching file paths."""
    from mcp_server.qdrant_client import ensure_collection, search_chunks
    with patch("mcp_server.qdrant_client.get_client", return_value=populated_client):
        ensure_collection(4)
        hits = search_chunks(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            limit=10,
            file_pattern=".py"
        )
    assert len(hits) == 1
    assert hits[0]["file_path"] == "main.py"


def test_collection_stats(populated_client):
    """Stats should report correct point count."""
    info = populated_client.get_collection(settings.qdrant_collection)
    assert info.points_count == 3


def test_empty_collection_search(memory_client):
    """Search on empty collection should return empty list."""
    response = memory_client.query_points(
        collection_name=settings.qdrant_collection,
        query=[0.1, 0.2, 0.3, 0.4],
        limit=5,
    )
    assert len(response.points) == 0


# ---------------------------------------------------------------------------
# delete_file_points tests (exercised directly against the in-memory client)
# ---------------------------------------------------------------------------


def _delete_file_points_impl(client, collection_name: str, file_path: str, directory: str) -> int:
    """Local replica of delete_file_points() logic, operating on an arbitrary client/collection."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="directory", match=MatchValue(value=directory)),
                FieldCondition(key="file_path", match=MatchValue(value=file_path)),
            ]
        ),
        limit=1,
    )
    if not points:
        return 0

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="directory", match=MatchValue(value=directory)),
                FieldCondition(key="file_path", match=MatchValue(value=file_path)),
            ]
        ),
    )
    return -1


def test_delete_file_points_removes_matching(populated_client):
    """delete_file_points should delete only points for the specified file."""
    result = _delete_file_points_impl(
        populated_client, settings.qdrant_collection, "main.py", "/test/project"
    )
    assert result == -1  # Indicates deletion occurred

    info = populated_client.get_collection(settings.qdrant_collection)
    assert info.points_count == 2  # main.py removed, app.js + queries.sql remain

    # Verify the remaining files are not main.py
    remaining, _ = populated_client.scroll(collection_name=settings.qdrant_collection, limit=10)
    remaining_paths = [p.payload["file_path"] for p in remaining]
    assert "main.py" not in remaining_paths
    assert "app.js" in remaining_paths
    assert "queries.sql" in remaining_paths


def test_delete_file_points_no_match_returns_zero(populated_client):
    """delete_file_points should return 0 when no points match."""
    result = _delete_file_points_impl(
        populated_client, settings.qdrant_collection, "nonexistent.py", "/test/project"
    )
    assert result == 0
    # Collection unchanged
    info = populated_client.get_collection(settings.qdrant_collection)
    assert info.points_count == 3


def test_delete_file_points_scoped_to_directory(populated_client):
    """delete_file_points with wrong directory should not delete the file's points."""
    # queries.sql belongs to /other/project — passing /test/project should be a no-op
    result = _delete_file_points_impl(
        populated_client, settings.qdrant_collection, "queries.sql", "/test/project"
    )
    assert result == 0
    info = populated_client.get_collection(settings.qdrant_collection)
    assert info.points_count == 3


def test_delete_file_points_multiple_chunks(memory_client):
    """delete_file_points should remove all chunks for a multi-chunk file."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={
                "text": f"chunk {i}",
                "file_path": "big_file.py",
                "directory": "/project",
                "chunk_index": i,
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        )
        for i in range(5)
    ] + [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.9, 0.8, 0.7, 0.6],
            payload={
                "text": "other file",
                "file_path": "other.py",
                "directory": "/project",
                "chunk_index": 0,
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        )
    ]
    memory_client.upsert(collection_name=settings.qdrant_collection, points=points)

    result = _delete_file_points_impl(
        memory_client, settings.qdrant_collection, "big_file.py", "/project"
    )
    assert result == -1

    info = memory_client.get_collection(settings.qdrant_collection)
    assert info.points_count == 1  # Only other.py remains
