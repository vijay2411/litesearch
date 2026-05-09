from __future__ import annotations

import sqlite3
from typing import List

from ..embedder import Embedder
from ..types import Candidate
from .bm25 import bm25_search
from .semantic import semantic_search

RRF_K = 60


def _rrf_fuse(
    bm25: List[Candidate], sem: List[Candidate]
) -> List[Candidate]:
    by_id: dict[int, Candidate] = {}
    fused: dict[int, float] = {}
    for rank, c in enumerate(bm25):
        fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id.setdefault(c.chunk_id, c)
    for rank, c in enumerate(sem):
        fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id[c.chunk_id] = c

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Candidate] = []
    for chunk_id, s in ordered:
        c = by_id[chunk_id]
        c.score = s
        out.append(c)
    return out


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    pool_size: int,
    embedder: Embedder,
    *,
    keep_vectors: bool = False,
) -> List[Candidate]:
    bm = bm25_search(conn, query, pool_size)
    sem = semantic_search(conn, query, pool_size, embedder, keep_vectors=keep_vectors)
    return _rrf_fuse(bm, sem)[:pool_size]
