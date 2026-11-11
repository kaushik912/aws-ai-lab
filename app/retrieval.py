"""Local hybrid retrieval: FAISS (dense) + BM25 (keyword), fused with Reciprocal Rank Fusion.

Replaces Azure AI Search. Index = two files in ./index (faiss.index, chunks.json) built by
scripts/ingest.py and shipped inside the Lambda image: zero idle cost, no cluster.
"""
import json
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_settings
from .llm import embed

RRF_K = 60  # standard constant; dampens the influence of top ranks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """rankings: lists of doc ids, best first. score = sum 1/(k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Index:
    faiss: faiss.Index
    chunks: list[dict]
    bm25: BM25Okapi

    @classmethod
    def load(cls, index_dir: str) -> "Index":
        d = pathlib.Path(index_dir)
        chunks = json.loads((d / "chunks.json").read_text())
        return cls(
            faiss=faiss.read_index(str(d / "faiss.index")),
            chunks=chunks,
            bm25=BM25Okapi([tokenize(c["content"]) for c in chunks]),
        )

    def search(self, query: str, qvec: np.ndarray, k: int, pool: int = 20) -> list[dict]:
        pool = min(pool, len(self.chunks))
        _, dense_ids = self.faiss.search(qvec[None, :], pool)
        dense = [int(i) for i in dense_ids[0] if i >= 0]
        bm = self.bm25.get_scores(tokenize(query))
        sparse = [int(i) for i in np.argsort(bm)[::-1][:pool] if bm[i] > 0]
        fused = rrf([dense, sparse])
        out = []
        for i in sorted(fused, key=fused.get, reverse=True)[:k]:
            # Normalised embeddings => inner product == cosine. Used for the grounding gate,
            # because RRF scores are rank-based and say nothing about absolute relevance.
            cos = float(np.dot(self.faiss.reconstruct(i), qvec))
            c = self.chunks[i]
            out.append(
                {"id": c["id"], "source": c["source"], "content": c["content"],
                 "rrf": round(fused[i], 5), "cosine": round(cos, 4)}
            )
        return out


@lru_cache
def get_index() -> Index:
    return Index.load(get_settings().index_dir)


async def hybrid_search(query: str, k: int | None = None) -> list[dict]:
    qvec = np.asarray(await embed(query), dtype="float32")
    return get_index().search(query, qvec, k or get_settings().top_k)
