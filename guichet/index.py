"""Retrieval: dense vectors, BM25 keywords, and the fusion of the two.

Dense and sparse retrieval fail in opposite directions, which is why both are
here. Embeddings handle paraphrase and cross-lingual matching but are weak on
exact identifiers — a query naming "VLS-TS" or "article L313-7" wants the
chunk containing that literal string, and a vector search will happily return
something merely topical. BM25 is the reverse: precise on rare tokens, blind to
the fact that "residence permit" and "titre de séjour" mean the same thing.

Reciprocal Rank Fusion combines the two rankings without needing the scores to
be on comparable scales, which they are not. Each document scores
sum(1 / (k + rank)) across the rankings it appears in, k=60 by convention.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from .chunk import Chunk
from .langid import detect
from .normalize import tokenize

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60

Mode = Literal["dense", "bm25", "hybrid", "hybrid-lang"]

# Weight given to the sparse ranking in "hybrid-lang" mode when the query is
# not in the corpus language. 0.0 drops BM25 entirely for those queries.
FOREIGN_SPARSE_WEIGHT = 0.0
CORPUS_LANGUAGE = "fr"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class DenseIndex:
    """Embeddings with an on-disk cache keyed by model and corpus content."""

    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL,
                 cache_dir: Path = Path("data/cache")):
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self.vectors = self._load_or_embed()

    def _cache_path(self) -> Path:
        fingerprint = hashlib.sha256(
            (self.model_name + "|".join(c.chunk_id for c in self.chunks)).encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"emb-{fingerprint}.npy"

    def _embedder(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._embedder().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)

    def _load_or_embed(self) -> np.ndarray:
        path = self._cache_path()
        if path.exists():
            return np.load(path)
        print(f"embedding {len(self.chunks)} chunks with {self.model_name}")
        vectors = self.encode([c.embed_text for c in self.chunks])
        np.save(path, vectors)
        return vectors

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_vector = self.encode([query])[0]
        scores = self.vectors @ query_vector
        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class BM25Index:
    """Keyword matching over the normalised, accent-folded token stream."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]], k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Weighted RRF. With no weights every ranking counts equally (plain RRF)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight == 0:
            continue
        for rank, (index, _score) in enumerate(ranking, start=1):
            fused[index] = fused.get(index, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda pair: -pair[1])


class Retriever:
    """The thing under test. Mode is swappable so the eval can compare them."""

    def __init__(self, chunks: list[Chunk], mode: Mode = "hybrid",
                 model_name: str = DEFAULT_MODEL,
                 foreign_sparse_weight: float = FOREIGN_SPARSE_WEIGHT):
        self.chunks = chunks
        self.mode = mode
        self.foreign_sparse_weight = foreign_sparse_weight
        uses_dense = mode in ("dense", "hybrid", "hybrid-lang")
        uses_sparse = mode in ("bm25", "hybrid", "hybrid-lang")
        self.dense = DenseIndex(chunks, model_name) if uses_dense else None
        self.sparse = BM25Index(chunks) if uses_sparse else None

    def search(self, query: str, top_k: int = 10, pool: int = 50) -> list[Hit]:
        if self.mode == "dense":
            ranked = self.dense.search(query, top_k)
        elif self.mode == "bm25":
            ranked = self.sparse.search(query, top_k)
        else:
            dense_hits = self.dense.search(query, pool)
            sparse_hits = self.sparse.search(query, pool)
            weights = [1.0, 1.0]
            if self.mode == "hybrid-lang" and detect(query) != CORPUS_LANGUAGE:
                # Out-of-language query: BM25 matches stray tokens ("France",
                # "residence") and plain RRF would promote that noise.
                weights = [1.0, self.foreign_sparse_weight]
            ranked = reciprocal_rank_fusion([dense_hits, sparse_hits],
                                            weights=weights)[:top_k]

        return [
            Hit(chunk=self.chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]


def build(chunks_path: Path = Path("data/corpus/chunks.jsonl"),
          mode: Mode = "hybrid", model_name: str = DEFAULT_MODEL) -> Retriever:
    from .chunk import load_chunks

    return Retriever(load_chunks(chunks_path), mode=mode, model_name=model_name)


if __name__ == "__main__":
    retriever = build()
    for query in [
        "How long is a récépissé valid?",
        "Combien de temps est valable un récépissé ?",
        "Can a foreign student work while studying in France?",
        "VLS-TS validation delay",
    ]:
        print(f"\n--- {query}")
        for hit in retriever.search(query, top_k=3):
            print(f"  {hit.rank}. [{hit.chunk.doc_id}] {hit.chunk.title[:58]}")
            print(f"     {hit.chunk.body[:110].replace(chr(10), ' ')}...")
