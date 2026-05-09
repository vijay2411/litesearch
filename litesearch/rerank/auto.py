from __future__ import annotations

import sqlite3
from typing import List

from ..types import Candidate
from .cross_encoder import rerank_cross_encoder
from .llm_judge import rerank_llm
from .mmr import rerank_mmr

RRF_K = 60


def rerank_auto(
    conn: sqlite3.Connection,
    query: str,
    query_vec: List[float],
    candidates: List[Candidate],
    top_k: int,
    *,
    dim: int = 768,
    mmr_lambda: float = 0.7,
    rerank_cli: str = "rerank",
    llm_backend: str = "gemini",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "gemma4",
) -> List[Candidate]:
    if not candidates:
        return []

    ce = rerank_cross_encoder(query, list(candidates), len(candidates), cli=rerank_cli)
    mm = rerank_mmr(conn, query_vec, list(candidates), len(candidates), dim=dim, lambda_=mmr_lambda)
    ll = rerank_llm(
        query, list(candidates), len(candidates),
        backend=llm_backend, gemini_api_key=gemini_api_key,
        gemini_model=gemini_model, ollama_url=ollama_url, ollama_model=ollama_model,
    )

    fused: dict[int, float] = {}
    by_id: dict[int, Candidate] = {}
    for ranking in (ce, mm, ll):
        for rank, c in enumerate(ranking):
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            by_id.setdefault(c.chunk_id, c)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Candidate] = []
    for chunk_id, s in ordered[:top_k]:
        c = by_id[chunk_id]
        c.score = s
        out.append(c)
    return out
