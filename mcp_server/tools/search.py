"""MCP tools for semantic and hybrid search over indexed content."""

import logging

from config.settings import settings
from mcp_server.embeddings import get_embedding
from mcp_server.ingestion import ingest_directory, needs_ingestion
from mcp_server.qdrant_client import search_chunks
from mcp_server.storage import BM25Index, reciprocal_rank_fusion

log = logging.getLogger("omni-rag")

# Global cache for BM25 indexes by directory
_bm25_cache: dict[str, BM25Index] = {}


def _get_bm25_index(directory: str) -> BM25Index | None:
    if directory in _bm25_cache:
        return _bm25_cache[directory]

    index = BM25Index(directory)
    if index.load():
        _bm25_cache[directory] = index
        return index
    return None


def search(
    query: str,
    n_results: int | None = None,
) -> str:
    """Search indexed files using semantic or hybrid search.

    Args:
        query: Natural language query.
        n_results: Optional number of results to return (default: 10).
    """
    try:
        directory = settings.get_working_directory()

        # Auto-ingest if needed
        if needs_ingestion(directory):
            log.info("Auto-ingesting %s before search", directory)
            ingest_directory(directory)
            # Clear cache to force reload after ingestion
            _bm25_cache.pop(directory, None)

        n = n_results or settings.default_n_results
        n = min(n, settings.max_n_results)

        # 1. Semantic search
        query_vector = get_embedding(query)
        semantic_results = search_chunks(query_vector, limit=n * 2)

        # 2. Hybrid search if enabled and BM25 index exists
        if settings.hybrid_search_enabled:
            bm25 = _get_bm25_index(directory)
            if bm25:
                bm25_results = bm25.search(query, top_k=n * 2)
                results = reciprocal_rank_fusion(
                    semantic_results,
                    bm25_results,
                    semantic_weight=settings.hybrid_semantic_weight,
                    bm25_weight=settings.hybrid_bm25_weight,
                )
                # Take top n
                results = results[:n]
            else:
                results = semantic_results[:n]
        else:
            results = semantic_results[:n]

        if not results:
            return f"No results found for '{query}' in {directory}"

        formatted = []
        for r in results:
            file_path = r.get("file_path", "unknown")
            text = r.get("text", "")
            score = r.get("fused_score") or r.get("score", 0.0)

            formatted.append(f"--- {file_path} (score: {score:.3f}) ---\n{text}\n")

        return "\n".join(formatted)
    except Exception as e:
        log.error("Error during search: %s", e)
        return f"Error during search: {e}"


def search_by_file(
    query: str,
    file_pattern: str,
    n_results: int | None = None,
) -> str:
    """Search indexed files, restricting results to paths matching a pattern.

    Args:
        query: Natural language query.
        file_pattern: Glob-style pattern or substring for file paths (e.g. '.py', 'models/').
        n_results: Optional number of results to return (default: 10).
    """
    try:
        directory = settings.get_working_directory()

        if needs_ingestion(directory):
            ingest_directory(directory)

        n = n_results or settings.default_n_results
        n = min(n, settings.max_n_results)

        query_vector = get_embedding(query)
        results = search_chunks(
            query_vector,
            limit=n,
            directory_filter=directory,
            file_pattern=file_pattern
        )

        if not results:
            return f"No results found for '{query}' matching file pattern '{file_pattern}' in {directory}"

        formatted = []
        for r in results:
            file_path = r.get("file_path", "unknown")
            text = r.get("text", "")
            score = r.get("score", 0.0)
            formatted.append(f"--- {file_path} (score: {score:.3f}) ---\n{text}\n")

        return "\n".join(formatted)
    except Exception as e:
        log.error("Error during file-filtered search: %s", e)
        return f"Error during search: {e}"
