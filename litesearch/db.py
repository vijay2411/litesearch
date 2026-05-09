from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id           INTEGER PRIMARY KEY,
  path         TEXT UNIQUE NOT NULL,
  title        TEXT,
  frontmatter  TEXT,
  content      TEXT NOT NULL,
  content_hash TEXT,
  mtime        INTEGER,
  created_at   INTEGER
);

CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  doc_id        INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   INTEGER,
  heading_path  TEXT,
  text          TEXT NOT NULL,
  text_hash     TEXT,
  start_line    INTEGER,
  end_line      INTEGER,
  doc_title     TEXT,
  doc_path      TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading_path, doc_title, doc_path,
  content='chunks', content_rowid='id',
  tokenize='porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, heading_path, doc_title, doc_path) VALUES (new.id, new.text, new.heading_path, new.doc_title, new.doc_path);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path, doc_title, doc_path) VALUES ('delete', old.id, old.text, old.heading_path, old.doc_title, old.doc_path);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path, doc_title, doc_path) VALUES ('delete', old.id, old.text, old.heading_path, old.doc_title, old.doc_path);
  INSERT INTO chunks_fts(rowid, text, heading_path, doc_title, doc_path) VALUES (new.id, new.text, new.heading_path, new.doc_title, new.doc_path);
END;
"""


def _vec_table_sql(dim: int) -> str:
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}]);"
    )


def connect(db_path: str | Path, dim: int = 768) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db(conn: sqlite3.Connection, dim: int = 768) -> None:
    _migrate_chunks_fts(conn)
    conn.executescript(SCHEMA)
    conn.execute(_vec_table_sql(dim))
    _rebuild_fts_if_needed(conn)
    conn.commit()


def _migrate_chunks_fts(conn: sqlite3.Connection) -> None:
    """Migrate chunks table and FTS5 to include doc_title and doc_path columns.
    Runs once on DBs created before this schema version."""
    cur = conn.cursor()
    try:
        cols = {r[1] for r in cur.execute("PRAGMA table_info(chunks)").fetchall()}
    except sqlite3.OperationalError:
        return
    if not cols or "doc_title" in cols:
        return
    cur.execute("ALTER TABLE chunks ADD COLUMN doc_title TEXT")
    cur.execute("ALTER TABLE chunks ADD COLUMN doc_path TEXT")
    cur.execute(
        "UPDATE chunks SET doc_title = (SELECT title FROM documents WHERE id = chunks.doc_id), "
        "doc_path = (SELECT path FROM documents WHERE id = chunks.doc_id)"
    )
    cur.execute("DROP TRIGGER IF EXISTS chunks_ai")
    cur.execute("DROP TRIGGER IF EXISTS chunks_ad")
    cur.execute("DROP TRIGGER IF EXISTS chunks_au")
    cur.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.commit()
    conn.execute("CREATE TABLE IF NOT EXISTS _fts_needs_rebuild (x INTEGER)")


def _rebuild_fts_if_needed(conn: sqlite3.Connection) -> None:
    """One-shot FTS5 rebuild after migration adds doc_title/doc_path columns."""
    try:
        conn.execute("SELECT 1 FROM _fts_needs_rebuild LIMIT 1")
    except sqlite3.OperationalError:
        return
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.execute("DROP TABLE _fts_needs_rebuild")
    conn.commit()
