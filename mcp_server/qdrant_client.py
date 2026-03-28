"""Qdrant vector database connection and operations."""

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config.settings import settings

log = logging.getLogger("codebase-rag-mcp")

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Lazy-loaded Qdrant client connection."""
    global _client
    if _client is not None:
        return _client

    if settings.qdrant_mode == "local":
        path = settings.get_qdrant_local_path()
        os.makedirs(path, exist_ok=True)
        log.info("Using local Qdrant at %s", path)
        _client = QdrantClient(path=path)
    else:
        log.info("Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    return _client


def ensure_collection(vector_size: int) -> None:
    """Create collection if it doesn't exist. Recreate if vector size changed."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection in collections:
        info = client.get_collection(settings.qdrant_collection)
        vectors = info.config.params.vectors
        existing_size = vectors.size if isinstance(vectors, VectorParams) else None
        if existing_size != vector_size:
            log.warning(
                "Vector dimension mismatch (existing=%d, new=%d). Recreating collection.",
                existing_size,
                vector_size,
            )
            client.delete_collection(settings.qdrant_collection)
        else:
            log.info(
                "Collection '%s' already exists (vector_size=%d)",
                settings.qdrant_collection,
                vector_size,
            )
            return

    log.info(
        "Creating collection '%s' (vector_size=%d)",
        settings.qdrant_collection,
        vector_size,
    )
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def is_directory_indexed(directory: str) -> bool:
    """Check if a directory has been ingested by querying metadata."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        return False

    results = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="directory", match=MatchValue(value=directory))]
        ),
        limit=1,
    )
    points, _ = results
    return len(points) > 0


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Store embedded chunks in Qdrant.

    Each chunk dict must have: id, text, file_path, directory, chunk_index, ingested_at
    Returns the number of points upserted.
    """
    client = get_client()

    points = [
        PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={
                "text": chunk["text"],
                "file_path": chunk["file_path"],
                "directory": chunk["directory"],
                "chunk_index": chunk["chunk_index"],
                "ingested_at": chunk["ingested_at"],
            },
        )
        for chunk, embedding in zip(chunks, embeddings, strict=False)
    ]

    # Upsert in batches of 100
    batch_size = 100
    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        total += len(batch)

    return total


def search(
    query_embedding: list[float],
    n_results: int,
    directory_filter: str | None = None,
    file_pattern: str | None = None,
) -> list[dict]:
    """Query Qdrant with optional directory and file pattern filters.

    Returns list of dicts with keys: text, file_path, score, directory.
    """
    client = get_client()

    must_conditions = []
    if directory_filter:
        must_conditions.append(
            FieldCondition(key="directory", match=MatchValue(value=directory_filter))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None  # type: ignore[arg-type]

    # If file_pattern is set, fetch extra results and filter client-side
    fetch_n = n_results * 5 if file_pattern else n_results

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        query_filter=query_filter,
        limit=fetch_n,
    )

    hits = []
    for point in response.points:
        payload = point.payload or {}
        file_path = payload.get("file_path", "unknown")

        if file_pattern and file_pattern.lower() not in file_path.lower():
            continue

        hits.append(
            {
                "text": payload.get("text", ""),
                "file_path": file_path,
                "score": point.score,
                "directory": payload.get("directory", ""),
            }
        )

        if len(hits) >= n_results:
            break

    return hits


def get_stats() -> dict:
    """Get collection statistics."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.qdrant_collection not in collections:
        return {
            "collection_name": settings.qdrant_collection,
            "exists": False,
            "total_points": 0,
        }

    info = client.get_collection(settings.qdrant_collection)
    stats = {
        "collection_name": settings.qdrant_collection,
        "exists": True,
        "total_points": info.points_count,
        "vectors_count": info.indexed_vectors_count,
        "status": str(info.status),
        "embedding_provider": settings.embedding_provider,
    }

    if settings.qdrant_mode == "local":
        stats["qdrant_mode"] = "local"
        stats["qdrant_path"] = settings.get_qdrant_local_path()
    else:
        stats["qdrant_mode"] = "remote"
        stats["qdrant_host"] = settings.qdrant_host
        stats["qdrant_port"] = settings.qdrant_port

    return stats


def delete_directory_points(directory: str) -> int:
    """Delete all points belonging to a directory. Returns count deleted."""
    client = get_client()
    # Get count before deletion
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="directory", match=MatchValue(value=directory))]
        ),
        limit=1,
    )
    if not points:
        return 0

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="directory", match=MatchValue(value=directory))]
        ),
    )
    return -1  # Qdrant doesn't return count; caller can re-check if needed


def delete_file_points(file_path: str, directory: str) -> int:
    """Delete all points belonging to a specific file. Returns count deleted.

    Args:
        file_path: Relative path to the file (e.g. 'mcp_server/ingestion.py').
        directory: Directory the file belongs to (used to scope the filter).
    """
    client = get_client()
    # Check whether any points exist before issuing a delete
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection,
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
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[
                FieldCondition(key="directory", match=MatchValue(value=directory)),
                FieldCondition(key="file_path", match=MatchValue(value=file_path)),
            ]
        ),
    )
    return -1  # Qdrant doesn't return count; caller can re-check if needed
