"""Tests for Qdrant client operations using in-memory Qdrant."""

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams


@pytest.fixture
def memory_client():
    """Create an in-memory Qdrant client for testing."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_codebase",
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
    memory_client.upsert(collection_name="test_codebase", points=points)
    return memory_client


def test_collection_creation(memory_client):
    """Collection should exist after creation."""
    collections = [c.name for c in memory_client.get_collections().collections]
    assert "test_codebase" in collections


def test_search_returns_results(populated_client):
    """Search should return matching results."""
    response = populated_client.query_points(
        collection_name="test_codebase",
        query=[0.1, 0.2, 0.3, 0.4],
        limit=2,
    )
    assert len(response.points) == 2
    assert response.points[0].payload["file_path"] == "main.py"


def test_search_with_filter(populated_client):
    """Search with directory filter should restrict results."""
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    response = populated_client.query_points(
        collection_name="test_codebase",
        query=[0.1, 0.2, 0.3, 0.4],
        query_filter=Filter(
            must=[FieldCondition(key="directory", match=MatchValue(value="/test/project"))]
        ),
        limit=10,
    )
    assert len(response.points) == 2
    for r in response.points:
        assert r.payload["directory"] == "/test/project"


def test_collection_stats(populated_client):
    """Stats should report correct point count."""
    info = populated_client.get_collection("test_codebase")
    assert info.points_count == 3


def test_empty_collection_search(memory_client):
    """Search on empty collection should return empty list."""
    response = memory_client.query_points(
        collection_name="test_codebase",
        query=[0.1, 0.2, 0.3, 0.4],
        limit=5,
    )
    assert len(response.points) == 0
