from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from .chunker import Chunk, chunk_body
from .embedder import Embedder, to_blob
from .parser import parse_markdown

log = logging.getLogger("litesearch.indexer")

_INDEX_LOCK = threading.Lock()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relpath(p: Path, vault: Path) -> str:
    try:
        return str(p.relative_to(vault))
    except ValueError:
        return str(p)


def _is_hidden(rel: str) -> bool:
    return any(part.startswith(".") for part in rel.split("/") if part)


def _embed_and_insert_chunks(
    conn: sqlite3.Connection,
    doc_id: int,
    chunks: List[Chunk],
    embedder: Embedder,
    doc_title: str,
    doc_path: str,
) -> None:
    cur = conn.cursor()

    existing = {
        row["text_hash"]: row["id"]
        for row in cur.execute(
            "SELECT id, text_hash FROM chunks WHERE doc_id = ?", (doc_id,)
        )
    }

    old_embeddings: dict[str, bytes] = {}
    for h, cid in existing.items():
        row = cur.execute(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row is not None:
            old_embeddings[h] = row["embedding"]

    if existing:
        ids = list(existing.values())
        placeholders = ",".join("?" * len(ids))
        cur.execute(
            f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})", ids
        )
    cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    # Hash includes title so that a title change triggers re-embedding.
    new_rows = []
    for idx, c in enumerate(chunks):
        h = _sha256(f"{doc_title}\n{c.text}")
        new_rows.append((idx, h, c))

    new_ids: List[int] = []
    for idx, h, c in new_rows:
        cur.execute(
            "INSERT INTO chunks (doc_id, chunk_index, heading_path, text, text_hash, "
            "start_line, end_line, doc_title, doc_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, idx, c.heading_path, c.text, h, c.start_line, c.end_line,
             doc_title, doc_path),
        )
        new_ids.append(cur.lastrowid)

    to_embed_idx = [i for i, (_, h, _) in enumerate(new_rows) if h not in old_embeddings]
    if to_embed_idx:
        # Prepend title to embedding text so semantic search covers filenames.
        texts = [f"{doc_title} | {new_rows[i][2].text}" for i in to_embed_idx]
        log.info("Embedding %d chunks for doc %d", len(texts), doc_id)
        vecs = embedder.embed_documents(texts)
        for j, i in enumerate(to_embed_idx):
            cur.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (new_ids[i], to_blob(vecs[j])),
            )

    for i, (_, h, _) in enumerate(new_rows):
        if h in old_embeddings:
            cur.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (new_ids[i], old_embeddings[h]),
            )


def index_file(
    conn: sqlite3.Connection,
    path: Path,
    embedder: Embedder,
    vault: Optional[Path] = None,
    soft_max_chars: int = 1500,
) -> bool:
    with _INDEX_LOCK:
        return _index_file_locked(conn, path, embedder, vault, soft_max_chars)


def _index_file_locked(
    conn: sqlite3.Connection,
    path: Path,
    embedder: Embedder,
    vault: Optional[Path],
    soft_max_chars: int,
) -> bool:
    vault = vault or path.parent
    rel = _relpath(path, vault)
    if _is_hidden(rel):
        return False

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False

    content_hash = _sha256(raw)
    cur = conn.cursor()
    row = cur.execute(
        "SELECT id, content_hash FROM documents WHERE path = ?", (rel,)
    ).fetchone()
    if row is not None and row["content_hash"] == content_hash:
        return False

    parsed = parse_markdown(path)
    mtime = int(path.stat().st_mtime)
    now = int(time.time())

    if row is None:
        cur.execute(
            "INSERT INTO documents (path, title, frontmatter, content, content_hash, mtime, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rel, parsed.title, parsed.frontmatter_json, raw, content_hash, mtime, now),
        )
        doc_id = cur.lastrowid
    else:
        doc_id = row["id"]
        cur.execute(
            "UPDATE documents SET title=?, frontmatter=?, content=?, content_hash=?, mtime=? WHERE id=?",
            (parsed.title, parsed.frontmatter_json, raw, content_hash, mtime, doc_id),
        )

    chunks = chunk_body(parsed.body, soft_max_chars)
    chunks = [c.with_offset(parsed.body_offset_lines) for c in chunks]
    _embed_and_insert_chunks(
        conn, doc_id, chunks, embedder,
        doc_title=parsed.title or rel, doc_path=rel,
    )

    conn.commit()
    return True


def index_text(
    conn: sqlite3.Connection,
    path: str,
    content: str,
    embedder: Embedder,
    title: Optional[str] = None,
    soft_max_chars: int = 1500,
) -> int:
    """Index raw text content directly (no file on disk needed).

    Returns the document id.
    """
    with _INDEX_LOCK:
        content_hash = _sha256(content)
        cur = conn.cursor()
        now = int(time.time())
        doc_title = title or path
        row = cur.execute(
            "SELECT id, content_hash FROM documents WHERE path = ?", (path,)
        ).fetchone()

        if row is not None and row["content_hash"] == content_hash:
            return row["id"]

        if row is None:
            cur.execute(
                "INSERT INTO documents (path, title, frontmatter, content, content_hash, mtime, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, doc_title, "{}", content, content_hash, now, now),
            )
            doc_id = cur.lastrowid
        else:
            doc_id = row["id"]
            cur.execute(
                "UPDATE documents SET title=?, content=?, content_hash=?, mtime=? WHERE id=?",
                (doc_title, content, content_hash, now, doc_id),
            )

        chunks = chunk_body(content, soft_max_chars)
        _embed_and_insert_chunks(
            conn, doc_id, chunks, embedder,
            doc_title=doc_title, doc_path=path,
        )
        conn.commit()
        return doc_id


def remove_doc(conn: sqlite3.Connection, path: str) -> bool:
    with _INDEX_LOCK:
        cur = conn.cursor()
        doc_row = cur.execute(
            "SELECT id FROM documents WHERE path = ?", (path,)
        ).fetchone()
        if doc_row is None:
            return False
        doc_id = doc_row["id"]
        chunk_ids = [
            r["id"]
            for r in cur.execute(
                "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        ]
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            cur.execute(
                f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    return True


def bootstrap_directory(
    conn: sqlite3.Connection,
    directory: Path,
    embedder: Embedder,
    soft_max_chars: int = 1500,
    glob: str = "*.md",
) -> int:
    if not directory.exists():
        log.warning("Directory does not exist: %s", directory)
        return 0

    seen = set()
    count_changed = 0
    for path in directory.rglob(glob):
        if not path.is_file():
            continue
        rel = _relpath(path, directory)
        if _is_hidden(rel):
            continue
        seen.add(rel)
        try:
            if index_file(conn, path, embedder, vault=directory, soft_max_chars=soft_max_chars):
                count_changed += 1
        except Exception as e:
            log.exception("Failed to index %s: %s", path, e)

    cur = conn.cursor()
    on_disk = list(seen)
    if on_disk:
        placeholders = ",".join("?" * len(on_disk))
        cur.execute(
            f"DELETE FROM documents WHERE path NOT IN ({placeholders})", on_disk
        )
    else:
        cur.execute("DELETE FROM documents")
    conn.commit()
    log.info("Bootstrap done. %d files indexed.", count_changed)
    return count_changed
