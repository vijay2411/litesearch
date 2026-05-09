"""CLI entry point: python -m litesearch <command>"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

INDEXES_DIR = Path.home() / ".litesearch"


def _resolve_db(name_or_path: str) -> str:
    """Resolve an index name to a .db path. If it looks like a path, use as-is."""
    if "/" in name_or_path or name_or_path.endswith(".db"):
        return name_or_path
    return str(INDEXES_DIR / f"{name_or_path}.db")


def _ensure_dir():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)


def _make_engine(args):
    from . import LiteSearch
    from .config import EmbeddingConfig, LiteSearchConfig

    db = _resolve_db(args.name)
    _ensure_dir()
    cfg = LiteSearchConfig(
        db_path=db,
        embedding=EmbeddingConfig(
            model=args.model, dimensions=args.dims,
            ollama_url=args.ollama_url,
        ),
    )
    return LiteSearch(config=cfg), db


def _add_embed_args(parser):
    parser.add_argument("--model", default="embeddinggemma", help="Ollama embedding model")
    parser.add_argument("--dims", type=int, default=768, help="Embedding dimensions")
    parser.add_argument("--ollama-url", default="http://localhost:11434")


def main():
    parser = argparse.ArgumentParser(
        prog="litesearch",
        description="Semantic search over SQLite. Manage named indexes from the command line.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- index ---
    idx = sub.add_parser("index", help="Index a directory into a named index")
    idx.add_argument("name", help="Index name (e.g. 'notes') or path to .db file")
    idx.add_argument("directory", help="Directory to index")
    idx.add_argument("--glob", default="*.*", help="File glob pattern (default: *.*)")
    _add_embed_args(idx)

    # --- index-jsonl ---
    jl = sub.add_parser("index-jsonl", help="Index a JSONL file (one doc per line)")
    jl.add_argument("name", help="Index name or path to .db file")
    jl.add_argument("file", help="Path to JSONL file")
    jl.add_argument("--text-field", default="text", help="JSON key for document text")
    jl.add_argument("--title-field", default="title", help="JSON key for document title")
    _add_embed_args(jl)

    # --- search ---
    q = sub.add_parser("search", help="Search an index")
    q.add_argument("name", help="Index name or path to .db file")
    q.add_argument("query", help="Search query")
    q.add_argument("--mode", default="semantic", choices=["grep", "bm25", "semantic", "hybrid"])
    q.add_argument("--reranker", default="none", choices=["none", "cross_encoder", "llm", "mmr", "auto"])
    q.add_argument("--top-k", type=int, default=10)
    _add_embed_args(q)

    # --- list ---
    sub.add_parser("list", help="List all named indexes")

    # --- info ---
    info = sub.add_parser("info", help="Show stats for an index")
    info.add_argument("name", help="Index name or path to .db file")

    # --- clear ---
    clr = sub.add_parser("clear", help="Delete all data from an index (keeps the file)")
    clr.add_argument("name", help="Index name or path to .db file")

    # --- delete ---
    dl = sub.add_parser("delete", help="Delete an index file entirely")
    dl.add_argument("name", help="Index name or path to .db file")

    # --- serve ---
    srv = sub.add_parser("serve", help="Start the REST API server for an index")
    srv.add_argument("name", help="Index name or path to .db file")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8900)
    _add_embed_args(srv)

    # --- legacy 'query' alias ---
    q2 = sub.add_parser("query", help=argparse.SUPPRESS)
    q2.add_argument("name", nargs="?", default="litesearch")
    q2.add_argument("query", help="Search query")
    q2.add_argument("--db", default=None)
    q2.add_argument("--mode", default="semantic", choices=["grep", "bm25", "semantic", "hybrid"])
    q2.add_argument("--reranker", default="none", choices=["none", "cross_encoder", "llm", "mmr", "auto"])
    q2.add_argument("--top-k", type=int, default=10)
    _add_embed_args(q2)

    args = parser.parse_args()

    if args.command == "index":
        engine, db = _make_engine(args)
        count = engine.add_directory(args.directory, glob=args.glob)
        print(f"Indexed {count} files → {db}")
        engine.close()

    elif args.command == "index-jsonl":
        engine, db = _make_engine(args)
        count = engine.add_jsonl(args.file, text_field=args.text_field, title_field=args.title_field)
        print(f"Indexed {count} documents → {db}")
        engine.close()

    elif args.command in ("search", "query"):
        if args.command == "query" and args.db:
            args.name = args.db
        engine, db = _make_engine(args)
        results = engine.search(args.query, mode=args.mode, reranker=args.reranker, top_k=args.top_k)
        if not results:
            print("No results.")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.score:.3f}] {r.doc_path}")
            if r.heading_path:
                print(f"   {r.heading_path}")
            snippet = r.snippet.replace("\n", " ")[:120]
            print(f"   {snippet}")
            print()
        engine.close()

    elif args.command == "list":
        _ensure_dir()
        dbs = sorted(INDEXES_DIR.glob("*.db"))
        if not dbs:
            print(f"No indexes found in {INDEXES_DIR}")
            return
        print(f"Indexes in {INDEXES_DIR}:\n")
        for db_path in dbs:
            name = db_path.stem
            size_mb = db_path.stat().st_size / (1024 * 1024)
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path))
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                conn.close()
                print(f"  {name:20s}  {doc_count:>6} docs  {chunk_count:>8} chunks  {size_mb:>7.1f} MB")
            except Exception:
                print(f"  {name:20s}  {size_mb:>7.1f} MB  (could not read)")
        print()

    elif args.command == "info":
        db = _resolve_db(args.name)
        if not Path(db).exists():
            print(f"Index not found: {db}")
            sys.exit(1)
        import sqlite3
        conn = sqlite3.connect(db)
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size_mb = Path(db).stat().st_size / (1024 * 1024)
        print(f"Index:    {args.name}")
        print(f"Path:     {db}")
        print(f"Size:     {size_mb:.1f} MB")
        print(f"Docs:     {doc_count}")
        print(f"Chunks:   {chunk_count}")
        recent = conn.execute(
            "SELECT path, title FROM documents ORDER BY mtime DESC LIMIT 5"
        ).fetchall()
        if recent:
            print(f"\nRecent documents:")
            for path, title in recent:
                print(f"  {path}  ({title})")
        conn.close()

    elif args.command == "clear":
        db = _resolve_db(args.name)
        if not Path(db).exists():
            print(f"Index not found: {db}")
            sys.exit(1)
        import sqlite3
        conn = sqlite3.connect(db)
        conn.execute("DELETE FROM chunks_vec")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
        print(f"Cleared all data from {args.name} ({db})")

    elif args.command == "delete":
        db = _resolve_db(args.name)
        if not Path(db).exists():
            print(f"Index not found: {db}")
            sys.exit(1)
        confirm = input(f"Delete {db}? [y/N] ").strip().lower()
        if confirm == "y":
            Path(db).unlink()
            print(f"Deleted {db}")
        else:
            print("Cancelled.")

    elif args.command == "serve":
        from .config import EmbeddingConfig, LiteSearchConfig, ServerConfig
        from .server import run

        db = _resolve_db(args.name)
        _ensure_dir()
        cfg = LiteSearchConfig(
            db_path=db,
            embedding=EmbeddingConfig(
                model=args.model, dimensions=args.dims,
                ollama_url=args.ollama_url,
            ),
            server=ServerConfig(host=args.host, port=args.port),
        )
        print(f"Serving {args.name} ({db}) on {args.host}:{args.port}")
        run(config=cfg)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
