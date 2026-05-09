from __future__ import annotations

import sqlite3
from typing import List

from ..embedder import Embedder, from_blob, to_blob
from ..types import Candidate


def _make_snippet(text: str, max_chars: int = 240) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    embedder: Embedder,
    *,
    keep_vectors: bool = False,
) -> List[Candidate]:
    if not query.strip():
        return []
    qvec = embedder.embed_query(query)
    qblob = to_blob(qvec)
    dim = embedder.dimensions

    rows = conn.execute(
        """
        SELECT v.chunk_id AS chunk_id, v.distance AS distance,
               c.doc_id, c.heading_path, c.text, c.start_line,
               d.path AS doc_path, d.title AS doc_title, d.mtime AS doc_mtime
          FROM chunks_vec v
          JOIN chunks c ON c.id = v.chunk_id
          JOIN documents d ON d.id = c.doc_id
         WHERE v.embedding MATCH ? AND k = ?
         ORDER BY v.distance
        """,
        (qblob, top_k),
    ).fetchall()

    out: List[Candidate] = []
    for r in rows:
        sim = 1.0 - (r["distance"] * r["distance"]) / 2.0
        sim = max(0.0, min(1.0, sim))

        vec = None
        if keep_vectors:
            vrow = conn.execute(
                "SELECT embedding FROM chunks_vec WHERE chunk_id = ?",
                (r["chunk_id"],),
            ).fetchone()
            if vrow is not None:
                vec = from_blob(vrow["embedding"], dim)

        out.append(
            Candidate(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_path=r["doc_path"],
                doc_title=r["doc_title"] or r["doc_path"],
                heading_path=r["heading_path"] or "",
                text=r["text"],
                snippet=_make_snippet(r["text"]),
                score=sim,
                doc_mtime=r["doc_mtime"] or 0,
                start_line=r["start_line"] or 1,
                vec=vec,
            )
        )
    return out


def query_vector(query: str, embedder: Embedder) -> List[float]:
    return embedder.embed_query(query)
