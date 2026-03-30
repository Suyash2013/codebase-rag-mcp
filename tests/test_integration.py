"""End-to-end integration tests using mock embeddings and in-memory Qdrant.

These tests verify the full pipeline: ingest -> embed -> store -> search.
Run with: python -m pytest tests/test_integration.py -v
Skip in fast CI with: python -m pytest -m "not integration"
"""

import hashlib
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient

from mcp_server.chunkers.recursive import RecursiveChunker

_rc = RecursiveChunker()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return [c.text for c in _rc.chunk(text, chunk_size, chunk_overlap)]


def _deterministic_embedding(text: str, dim: int = 384) -> list[float]:
    """Generate a deterministic embedding from text using hashing.

    This produces consistent vectors for the same input so search results
    are reproducible. Similar texts will NOT produce similar vectors (unlike
    a real model), but this is sufficient for testing the pipeline plumbing.
    """
    h = hashlib.sha256(text.encode()).digest()
    # Extend hash bytes to fill the dimension
    raw = h * ((dim * 4 // len(h)) + 1)
    values = []
    for i in range(dim):
        byte_val = raw[i % len(raw)]
        values.append((byte_val - 128) / 128.0)
    # Normalise
    norm = sum(v * v for v in values) ** 0.5
    if norm > 0:
        values = [v / norm for v in values]
    return values


@pytest.mark.integration
class TestIntegrationPipeline:
    """End-to-end tests for the ingest-search pipeline."""

    @pytest.fixture(autouse=True)
    def setup_environment(self, tmp_codebase):
        """Set up a temporary codebase and in-memory Qdrant for each test."""
        self.codebase_dir = str(tmp_codebase)
        self.tmp_codebase = tmp_codebase
        self.memory_client = QdrantClient(":memory:")

    def _create_patches(self):
        """Create patches for integration tests, returning (settings, patches_list)."""
        import os

        from config.settings import Settings

        patches = []

        env_patch = patch.dict(
            os.environ,
            {
                "RAG_QDRANT_MODE": "local",
                "RAG_EMBEDDING_PROVIDER": "onnx",
                "RAG_WORKING_DIRECTORY": self.codebase_dir,
                "RAG_QDRANT_COLLECTION": "test_integration",
                "RAG_CHUNK_SIZE": "500",
                "RAG_CHUNK_OVERLAP": "50",
            },
        )
        env_patch.start()
        patches.append(env_patch)

        test_settings = Settings()

        client_patch = patch(
            "mcp_server.qdrant_client.get_client",
            return_value=self.memory_client,
        )
        client_patch.start()
        patches.append(client_patch)

        settings_patch = patch("mcp_server.qdrant_client.settings", test_settings)
        settings_patch.start()
        patches.append(settings_patch)

        return test_settings, patches

    @staticmethod
    def _stop_patches(patches):
        for p in reversed(patches):
            p.stop()

    def test_ingest_and_search(self):
        """Ingest a small codebase and verify search returns results."""
        _test_settings, patches = self._create_patches()
        try:
            from mcp_server.ingestion import _collect_text_files
            from mcp_server.qdrant_client import ensure_collection, search, upsert_chunks

            dim = 384
            ensure_collection(dim)

            # Collect and chunk files
            files = _collect_text_files(self.codebase_dir)
            assert len(files) > 0

            # Embed chunks and store
            all_chunks = []
            all_embeddings = []
            now = datetime.now(timezone.utc).isoformat()

            for filepath, rel_path in files:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                chunks = _chunk_text(text, 500, 50)
                for i, chunk_text in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    all_chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "file_path": rel_path,
                            "directory": self.codebase_dir,
                            "chunk_index": i,
                            "ingested_at": now,
                        }
                    )
                    all_embeddings.append(_deterministic_embedding(chunk_text, dim))

            assert len(all_chunks) > 0
            count = upsert_chunks(all_chunks, all_embeddings)
            assert count == len(all_chunks)

            # Search and verify results
            query_embedding = _deterministic_embedding("hello world function", dim)
            hits = search(
                query_embedding,
                n_results=5,
                directory_filter=self.codebase_dir,
            )

            assert len(hits) > 0
            # Each hit should have expected fields
            for hit in hits:
                assert "text" in hit
                assert "file_path" in hit
                assert "score" in hit
                assert isinstance(hit["score"], float)
        finally:
            self._stop_patches(patches)

    def test_search_with_file_pattern(self):
        """Search with file_pattern should only return matching files."""
        _test_settings, patches = self._create_patches()
        try:
            from mcp_server.ingestion import _collect_text_files
            from mcp_server.qdrant_client import ensure_collection, search, upsert_chunks

            dim = 384
            ensure_collection(dim)

            files = _collect_text_files(self.codebase_dir)
            all_chunks = []
            all_embeddings = []
            now = datetime.now(timezone.utc).isoformat()

            for filepath, rel_path in files:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                chunks = _chunk_text(text, 500, 50)
                for i, chunk_text in enumerate(chunks):
                    chunk_id = str(uuid.uuid4())
                    all_chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "file_path": rel_path,
                            "directory": self.codebase_dir,
                            "chunk_index": i,
                            "ingested_at": now,
                        }
                    )
                    all_embeddings.append(_deterministic_embedding(chunk_text, dim))

            upsert_chunks(all_chunks, all_embeddings)

            # Search with file pattern restricting to .py files
            query_embedding = _deterministic_embedding("format name function", dim)
            hits = search(
                query_embedding,
                n_results=5,
                directory_filter=self.codebase_dir,
                file_pattern=".py",
            )

            # All results should match the file pattern
            for hit in hits:
                assert ".py" in hit["file_path"].lower()
        finally:
            self._stop_patches(patches)

    def test_ingest_empty_directory(self):
        """Ingesting an empty directory should return no files."""
        import tempfile

        from mcp_server.ingestion import _collect_text_files

        with tempfile.TemporaryDirectory() as empty_dir:
            files = _collect_text_files(empty_dir)
            assert len(files) == 0

    def test_delete_and_recount(self):
        """After deleting directory points, count should decrease."""
        _test_settings, patches = self._create_patches()
        try:
            from mcp_server.ingestion import _collect_text_files
            from mcp_server.qdrant_client import (
                delete_directory_points,
                ensure_collection,
                upsert_chunks,
            )

            dim = 384
            ensure_collection(dim)

            files = _collect_text_files(self.codebase_dir)
            all_chunks = []
            all_embeddings = []
            now = datetime.now(timezone.utc).isoformat()

            for filepath, rel_path in files:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                chunks = _chunk_text(text, 500, 50)
                for i, chunk_text in enumerate(chunks):
                    all_chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "file_path": rel_path,
                            "directory": self.codebase_dir,
                            "chunk_index": i,
                            "ingested_at": now,
                        }
                    )
                    all_embeddings.append(_deterministic_embedding(chunk_text, dim))

            initial_count = upsert_chunks(all_chunks, all_embeddings)
            assert initial_count > 0

            deleted = delete_directory_points(self.codebase_dir)
            assert deleted == initial_count

            # Verify collection is now empty for this directory
            info = self.memory_client.get_collection("test_integration")
            assert info.points_count == 0
        finally:
            self._stop_patches(patches)
