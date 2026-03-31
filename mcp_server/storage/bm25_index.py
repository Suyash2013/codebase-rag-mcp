import json
import logging
import math
import re
from pathlib import Path

log = logging.getLogger("omni-rag")

DATA_DIR = ".omni-rag"


class BM25Index:
    """Lightweight BM25-Okapi implementation. No external dependencies."""

    def __init__(self, directory: str, k1: float = 1.5, b: float = 0.75):
        self.directory = directory
        self.k1 = k1
        self.b = b
        self._corpus_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._n_docs: int = 0

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks. Each chunk has 'id' and 'text'."""
        self._corpus_ids = [c["id"] for c in chunks]
        self._corpus_tokens = [self._tokenize(c["text"]) for c in chunks]
        self._compute_stats()

    def update(self, add: list[dict] | None = None, remove: list[str] | None = None) -> None:
        """Incrementally update the corpus."""
        new_ids = list(self._corpus_ids)
        new_tokens = list(self._corpus_tokens)

        if remove:
            remove_set = set(remove)
            temp_ids = []
            temp_tokens = []
            for cid, tokens in zip(new_ids, new_tokens, strict=False):
                if cid not in remove_set:
                    temp_ids.append(cid)
                    temp_tokens.append(tokens)
            new_ids = temp_ids
            new_tokens = temp_tokens

        if add:
            for c in add:
                new_ids.append(c["id"])
                new_tokens.append(self._tokenize(c["text"]))

        self._corpus_ids = new_ids
        self._corpus_tokens = new_tokens
        self._compute_stats()

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs, sorted by score descending."""
        if not self._corpus_ids:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._corpus_tokens):
            score = 0.0
            doc_len = self._doc_lens[i]
            tf_map: dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = tf_map.get(qt, 0)
                idf = self._idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_dl)
                score += idf * numerator / denominator

            if score > 0:
                scores.append((self._corpus_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self) -> None:
        data_path = Path(self.directory) / DATA_DIR
        data_path.mkdir(parents=True, exist_ok=True)

        corpus = [{"id": cid, "tokens": tokens}
                  for cid, tokens in zip(self._corpus_ids, self._corpus_tokens, strict=False)]
        (data_path / "bm25_corpus.json").write_text(json.dumps(corpus))

    def load(self) -> bool:
        corpus_path = Path(self.directory) / DATA_DIR / "bm25_corpus.json"
        if not corpus_path.exists():
            return False
        try:
            corpus = json.loads(corpus_path.read_text())
            self._corpus_ids = [c["id"] for c in corpus]
            self._corpus_tokens = [c["tokens"] for c in corpus]
            self._compute_stats()
            return True
        except Exception as e:
            log.warning("Failed to load BM25 index: %s", e)
            return False

    def _compute_stats(self) -> None:
        self._n_docs = len(self._corpus_ids)
        self._doc_lens = [len(t) for t in self._corpus_tokens]
        self._avg_dl = sum(self._doc_lens) / self._n_docs if self._n_docs else 0

        self._doc_freqs = {}
        for tokens in self._corpus_tokens:
            seen = set(tokens)
            for t in seen:
                self._doc_freqs[t] = self._doc_freqs.get(t, 0) + 1

        self._idf = {}
        for term, df in self._doc_freqs.items():
            self._idf[term] = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        return re.findall(r"\w+", text.lower())
