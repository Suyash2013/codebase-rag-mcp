"""Unit tests for BM25Index and reciprocal_rank_fusion."""

import json
import math
import tempfile
from pathlib import Path

from mcp_server.storage.bm25_index import BM25Index
from mcp_server.storage.hybrid import reciprocal_rank_fusion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str) -> dict:
    return {"id": chunk_id, "text": text}


# ---------------------------------------------------------------------------
# BM25Index -tokenizer
# ---------------------------------------------------------------------------


class TestBM25Tokenizer:
    def setup_method(self):
        self.idx = BM25Index(directory="/tmp")

    def test_lowercases(self):
        assert self.idx._tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = self.idx._tokenize("foo, bar! baz.")
        assert tokens == ["foo", "bar", "baz"]

    def test_empty_string(self):
        assert self.idx._tokenize("") == []

    def test_numbers_kept(self):
        assert "42" in self.idx._tokenize("answer is 42")

    def test_underscore_kept(self):
        # \w includes underscore
        tokens = self.idx._tokenize("snake_case")
        assert "snake_case" in tokens


# ---------------------------------------------------------------------------
# BM25Index -build & basic invariants
# ---------------------------------------------------------------------------


class TestBM25Build:
    def setup_method(self):
        self.idx = BM25Index(directory="/tmp")

    def test_empty_build_returns_empty_search(self):
        self.idx.build([])
        assert self.idx.search("anything") == []

    def test_single_doc_returns_it(self):
        self.idx.build([_chunk("a", "hello world")])
        results = self.idx.search("hello")
        assert len(results) == 1
        assert results[0][0] == "a"
        assert results[0][1] > 0

    def test_correct_ids_stored(self):
        chunks = [_chunk("x", "foo"), _chunk("y", "bar")]
        self.idx.build(chunks)
        assert self.idx._corpus_ids == ["x", "y"]

    def test_n_docs_set_correctly(self):
        self.idx.build([_chunk("a", "hello"), _chunk("b", "world")])
        assert self.idx._n_docs == 2

    def test_avg_dl_positive(self):
        self.idx.build([_chunk("a", "one two three"), _chunk("b", "four five")])
        assert self.idx._avg_dl > 0


# ---------------------------------------------------------------------------
# BM25Index -search ranking
# ---------------------------------------------------------------------------


class TestBM25Search:
    def setup_method(self):
        self.idx = BM25Index(directory="/tmp")
        self.idx.build(
            [
                _chunk("python", "python programming language tutorial"),
                _chunk("java", "java programming language enterprise"),
                _chunk("rust", "rust systems programming memory safety"),
            ]
        )

    def test_relevant_doc_ranked_first(self):
        results = self.idx.search("rust memory")
        assert results[0][0] == "rust"

    def test_returns_scores_descending(self):
        results = self.idx.search("programming")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self):
        results = self.idx.search("programming", top_k=2)
        assert len(results) <= 2

    def test_no_match_returns_empty(self):
        results = self.idx.search("zzznomatchzzz")
        assert results == []

    def test_scores_are_positive(self):
        for _, score in self.idx.search("python"):
            assert score > 0

    def test_query_on_empty_index(self):
        idx = BM25Index(directory="/tmp")
        idx.build([])
        assert idx.search("hello") == []


# ---------------------------------------------------------------------------
# BM25Index -IDF calculation
# ---------------------------------------------------------------------------


class TestBM25IDF:
    def test_idf_rare_term_higher_than_common(self):
        idx = BM25Index(directory="/tmp")
        idx.build(
            [
                _chunk("a", "common word here"),
                _chunk("b", "common term here"),
                _chunk("c", "rare xyzzy here"),
            ]
        )
        # "common" appears in 2/3 docs; "xyzzy" appears in 1/3 docs
        assert idx._idf["xyzzy"] > idx._idf["common"]

    def test_idf_uses_okapi_formula(self):
        idx = BM25Index(directory="/tmp")
        idx.build([_chunk("a", "foo"), _chunk("b", "foo")])
        # With n=2, df=2: log((2-2+0.5)/(2+0.5)+1) = log(0.2+1) ≈ 0.182
        expected = math.log((2 - 2 + 0.5) / (2 + 0.5) + 1)
        assert abs(idx._idf["foo"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# BM25Index -update
# ---------------------------------------------------------------------------


class TestBM25Update:
    def setup_method(self):
        self.idx = BM25Index(directory="/tmp")
        self.idx.build(
            [
                _chunk("a", "hello world"),
                _chunk("b", "goodbye world"),
            ]
        )

    def test_add_new_doc(self):
        self.idx.update(add=[_chunk("c", "brand new content")])
        assert "c" in self.idx._corpus_ids
        assert self.idx._n_docs == 3

    def test_remove_doc(self):
        self.idx.update(remove=["a"])
        assert "a" not in self.idx._corpus_ids
        assert self.idx._n_docs == 1

    def test_add_and_remove_together(self):
        self.idx.update(add=[_chunk("c", "new doc")], remove=["b"])
        assert "b" not in self.idx._corpus_ids
        assert "c" in self.idx._corpus_ids
        assert self.idx._n_docs == 2

    def test_remove_nonexistent_is_noop(self):
        self.idx.update(remove=["does_not_exist"])
        assert self.idx._n_docs == 2

    def test_add_none_and_remove_none_is_noop(self):
        self.idx.update(add=None, remove=None)
        assert self.idx._n_docs == 2

    def test_updated_index_searchable(self):
        self.idx.update(add=[_chunk("c", "python snake")])
        results = self.idx.search("snake")
        ids = [r[0] for r in results]
        assert "c" in ids


# ---------------------------------------------------------------------------
# BM25Index -save / load
# ---------------------------------------------------------------------------


class TestBM25Persistence:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = BM25Index(directory=tmp)
            idx.build([_chunk("a", "hello world")])
            idx.save()
            assert (Path(tmp) / ".omni-rag" / "bm25_corpus.json").exists()

    def test_load_restores_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = BM25Index(directory=tmp)
            idx.build([_chunk("a", "hello world"), _chunk("b", "foo bar")])
            idx.save()

            idx2 = BM25Index(directory=tmp)
            loaded = idx2.load()

            assert loaded is True
            assert idx2._corpus_ids == ["a", "b"]
            assert idx2._n_docs == 2

    def test_load_returns_false_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = BM25Index(directory=tmp)
            assert idx.load() is False

    def test_load_then_search_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = BM25Index(directory=tmp)
            idx.build(
                [
                    _chunk("py", "python programming"),
                    _chunk("js", "javascript browser"),
                ]
            )
            idx.save()

            idx2 = BM25Index(directory=tmp)
            idx2.load()
            results = idx2.search("javascript")
            assert results[0][0] == "js"

    def test_save_stores_tokens_not_raw_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = BM25Index(directory=tmp)
            idx.build([_chunk("a", "Hello World!")])
            idx.save()
            data = json.loads((Path(tmp) / ".omni-rag" / "bm25_corpus.json").read_text())
            assert data[0]["tokens"] == ["hello", "world"]

    def test_load_handles_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / ".omni-rag"
            data_path.mkdir(parents=True)
            (data_path / "bm25_corpus.json").write_text("not valid json{{")
            idx = BM25Index(directory=tmp)
            assert idx.load() is False


# ---------------------------------------------------------------------------
# BM25Index -k1 / b parameters
# ---------------------------------------------------------------------------


class TestBM25Parameters:
    def test_custom_k1_b(self):
        idx = BM25Index(directory="/tmp", k1=1.2, b=0.5)
        assert idx.k1 == 1.2
        assert idx.b == 0.5

    def test_b_zero_ignores_doc_length(self):
        # b=0 removes document-length normalization: two docs with the same TF
        # for the query term but different total lengths should score identically.
        idx = BM25Index(directory="/tmp", k1=1.5, b=0.0)
        idx.build(
            [
                _chunk("a", "python foo bar baz"),  # tf("python")=1, len=4
                _chunk("b", "python alpha beta gamma delta"),  # tf("python")=1, len=5
            ]
        )
        results = idx.search("python")
        scores = {r[0]: r[1] for r in results}
        # denominator = tf + k1*(1 - 0 + 0*dl/avg_dl) = tf + k1 — same for both
        assert abs(scores["a"] - scores["b"]) < 1e-9


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_returns_all_unique_ids(self):
        semantic = [{"id": "a"}, {"id": "b"}]
        bm25 = [("c", 0.9), ("d", 0.5)]
        results = reciprocal_rank_fusion(semantic, bm25)
        ids = {r["id"] for r in results}
        assert ids == {"a", "b", "c", "d"}

    def test_result_has_fused_score(self):
        semantic = [{"id": "a"}]
        bm25 = [("a", 1.0)]
        results = reciprocal_rank_fusion(semantic, bm25)
        assert "fused_score" in results[0]

    def test_sorted_by_fused_score_descending(self):
        semantic = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        bm25 = [("c", 0.9), ("b", 0.5), ("a", 0.1)]
        results = reciprocal_rank_fusion(semantic, bm25)
        scores = [r["fused_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_overlap_boosts_score(self):
        # Document present in both rankings should outscore docs only in one
        semantic = [{"id": "overlap"}, {"id": "sem_only"}]
        bm25 = [("overlap", 1.0), ("bm25_only", 0.5)]
        results = reciprocal_rank_fusion(semantic, bm25)
        score_map = {r["id"]: r["fused_score"] for r in results}
        assert score_map["overlap"] > score_map["sem_only"]
        assert score_map["overlap"] > score_map["bm25_only"]

    def test_empty_inputs(self):
        results = reciprocal_rank_fusion([], [])
        assert results == []

    def test_only_semantic(self):
        semantic = [{"id": "a"}, {"id": "b"}]
        results = reciprocal_rank_fusion(semantic, [])
        assert len(results) == 2

    def test_only_bm25(self):
        bm25 = [("x", 1.0), ("y", 0.5)]
        results = reciprocal_rank_fusion([], bm25)
        assert len(results) == 2

    def test_weights_respected(self):
        # Heavily semantic-weighted: top semantic doc should beat top bm25-only doc
        semantic = [{"id": "sem_top"}]
        bm25 = [("bm25_top", 1.0)]
        results = reciprocal_rank_fusion(semantic, bm25, semantic_weight=0.9, bm25_weight=0.1)
        score_map = {r["id"]: r["fused_score"] for r in results}
        assert score_map["sem_top"] > score_map["bm25_top"]

    def test_result_map_preserves_metadata(self):
        semantic = [{"id": "a", "file_path": "foo.py", "score": 0.9}]
        bm25 = [("a", 1.0)]
        results = reciprocal_rank_fusion(semantic, bm25)
        assert results[0]["file_path"] == "foo.py"

    def test_fallback_id_from_file_path(self):
        # semantic results that use file_path instead of id
        semantic = [{"file_path": "bar.py"}]
        bm25 = []
        results = reciprocal_rank_fusion(semantic, bm25)
        assert len(results) == 1
        assert results[0]["fused_score"] > 0
