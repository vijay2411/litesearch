from __future__ import annotations

import re
import sqlite3
from typing import List

from ..types import Candidate

_FTS_SAFE = re.compile(r'[^\w\s"*]', re.UNICODE)


def _to_fts_query(q: str) -> str:
    cleaned = _FTS_SAFE.sub(" ", q).strip()
    if not cleaned:
        return ""
    if any(c in cleaned for c in ('"', "*")):
        return cleaned
    terms = [t for t in cleaned.split() if t]
    return " OR ".join(terms)


def bm25_search(conn: sqlite3.Connection, query: str, top_k: int) -> List[Candidate]:
    fts_q = _to_fts_query(query)
    if not fts_q:
        return []
    # FTS5 columns: 0=text, 1=heading_path, 2=doc_title, 3=doc_path
    # Weights: text=10, heading_path=5, doc_title=8, doc_path=3
    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.doc_id, c.heading_path, c.text, c.start_line,
               d.path AS doc_path, d.title AS doc_title, d.mtime AS doc_mtime,
               snippet(chunks_fts, 0, '<b>', '</b>', '…', 24) AS snippet,
               bm25(chunks_fts, 10.0, 5.0, 8.0, 3.0) AS rank
          FROM chunks_fts
          JOIN chunks c ON c.id = chunks_fts.rowid
          JOIN documents d ON d.id = c.doc_id
         WHERE chunks_fts MATCH ?
         ORDER BY bm25(chunks_fts, 10.0, 5.0, 8.0, 3.0)
         LIMIT ?
        """,
        (fts_q, top_k),
    ).fetchall()

    raw = [-row["rank"] for row in rows]
    if raw:
        lo, hi = min(raw), max(raw)
        span = hi - lo if hi > lo else 1.0

    out: List[Candidate] = []
    for i, r in enumerate(rows):
        score = (raw[i] - lo) / span if raw else 0.0
        out.append(
            Candidate(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_path=r["doc_path"],
                doc_title=r["doc_title"] or r["doc_path"],
                heading_path=r["heading_path"] or "",
                text=r["text"],
                snippet=r["snippet"] or "",
                score=score,
                doc_mtime=r["doc_mtime"] or 0,
                start_line=r["start_line"] or 1,
            )
        )
    return out
