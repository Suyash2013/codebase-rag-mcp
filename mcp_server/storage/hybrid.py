"""Hybrid search ranking fusion."""


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    """Fuse semantic and BM25 rankings using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    # Semantic ranking
    for rank, result in enumerate(semantic_results):
        rid = result.get("id") or result.get("file_path", str(rank))
        scores[rid] = scores.get(rid, 0) + semantic_weight / (k + rank + 1)
        result_map[rid] = result

    # BM25 ranking
    for rank, (chunk_id, bm25_score) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + bm25_weight / (k + rank + 1)
        if chunk_id not in result_map:
            result_map[chunk_id] = {"id": chunk_id}

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for rid in sorted_ids:
        entry = dict(result_map[rid])
        entry["fused_score"] = scores[rid]
        results.append(entry)

    return results
