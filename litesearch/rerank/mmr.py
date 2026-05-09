from __future__ import annotations

import math
import sqlite3
import struct
from typing import List

from ..types import Candidate


def _cos(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    return max(-1.0, min(1.0, dot))


def _ensure_vec(conn: sqlite3.Connection, c: Candidate, dim: int) -> List[float] | None:
    if c.vec is not None:
        return c.vec
    if c.chunk_id < 0:
        return None
    row = conn.execute(
        "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (c.chunk_id,)
    ).fetchone()
    if row is None:
        return None
    c.vec = list(struct.unpack(f"{dim}f", row["embedding"]))
    return c.vec


def rerank_mmr(
    conn: sqlite3.Connection,
    query_vec: List[float],
    candidates: List[Candidate],
    top_k: int,
    dim: int = 768,
    lambda_: float = 0.7,
) -> List[Candidate]:
    if not candidates:
        return []

    pool = [c for c in candidates if _ensure_vec(conn, c, dim) is not None]
    no_vec = [c for c in candidates if c not in pool]

    selected: List[Candidate] = []
    remaining = list(pool)
    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -math.inf
        for i, c in enumerate(remaining):
            relevance = _cos(query_vec, c.vec)
            if selected:
                diversity = max(_cos(c.vec, s.vec) for s in selected)
            else:
                diversity = 0.0
            score = lambda_ * relevance - (1 - lambda_) * diversity
            if score > best_score:
                best_score = score
                best_idx = i
        chosen = remaining.pop(best_idx)
        chosen.score = float(best_score)
        selected.append(chosen)

    if len(selected) < top_k:
        for c in no_vec:
            selected.append(c)
            if len(selected) >= top_k:
                break
    return selected
