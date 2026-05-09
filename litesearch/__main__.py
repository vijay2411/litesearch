"""CLI entry point: python -m litesearch <command>"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="litesearch", description="Pluggable semantic search over SQLite")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the REST API server")
    serve.add_argument("--db", default="litesearch.db", help="Path to SQLite database")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8900)
    serve.add_argument("--model", default="embeddinggemma", help="Ollama embedding model")
    serve.add_argument("--dims", type=int, default=768, help="Embedding dimensions")
    serve.add_argument("--ollama-url", default="http://localhost:11434")

    idx = sub.add_parser("index", help="Index a directory of files")
    idx.add_argument("directory", help="Directory to index")
    idx.add_argument("--db", default="litesearch.db")
    idx.add_argument("--glob", default="*.*", help="File glob pattern (default: *.*)")
    idx.add_argument("--model", default="embeddinggemma")
    idx.add_argument("--dims", type=int, default=768)
    idx.add_argument("--ollama-url", default="http://localhost:11434")

    jl = sub.add_parser("index-jsonl", help="Index a JSONL file (one doc per line)")
    jl.add_argument("file", help="Path to JSONL file")
    jl.add_argument("--db", default="litesearch.db")
    jl.add_argument("--text-field", default="text", help="JSON key for document text")
    jl.add_argument("--title-field", default="title", help="JSON key for document title")
    jl.add_argument("--model", default="embeddinggemma")
    jl.add_argument("--dims", type=int, default=768)
    jl.add_argument("--ollama-url", default="http://localhost:11434")

    q = sub.add_parser("query", help="Run a search query")
    q.add_argument("query", help="Search query")
    q.add_argument("--db", default="litesearch.db")
    q.add_argument("--mode", default="hybrid", choices=["grep", "bm25", "semantic", "hybrid"])
    q.add_argument("--reranker", default="none", choices=["none", "cross_encoder", "llm", "mmr", "auto"])
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--model", default="embeddinggemma")
    q.add_argument("--dims", type=int, default=768)
    q.add_argument("--ollama-url", default="http://localhost:11434")

    args = parser.parse_args()

    if args.command == "serve":
        from .config import EmbeddingConfig, LiteSearchConfig, ServerConfig
        from .server import run

        cfg = LiteSearchConfig(
            db_path=args.db,
            embedding=EmbeddingConfig(model=args.model, dimensions=args.dims, ollama_url=args.ollama_url),
            server=ServerConfig(host=args.host, port=args.port),
        )
        run(config=cfg)

    elif args.command == "index":
        from . import LiteSearch
        from .config import EmbeddingConfig, LiteSearchConfig
        from .embedder import OllamaEmbedder

        cfg = LiteSearchConfig(
            db_path=args.db,
            embedding=EmbeddingConfig(model=args.model, dimensions=args.dims, ollama_url=args.ollama_url),
        )
        engine = LiteSearch(config=cfg)
        count = engine.add_directory(args.directory, glob=args.glob)
        print(f"Indexed {count} files into {args.db}")

    elif args.command == "index-jsonl":
        from . import LiteSearch
        from .config import EmbeddingConfig, LiteSearchConfig

        cfg = LiteSearchConfig(
            db_path=args.db,
            embedding=EmbeddingConfig(model=args.model, dimensions=args.dims, ollama_url=args.ollama_url),
        )
        engine = LiteSearch(config=cfg)
        count = engine.add_jsonl(args.file, text_field=args.text_field, title_field=args.title_field)
        print(f"Indexed {count} documents from {args.file} into {args.db}")

    elif args.command == "query":
        from . import LiteSearch
        from .config import EmbeddingConfig, LiteSearchConfig

        cfg = LiteSearchConfig(
            db_path=args.db,
            embedding=EmbeddingConfig(model=args.model, dimensions=args.dims, ollama_url=args.ollama_url),
        )
        engine = LiteSearch(config=cfg)
        results = engine.search(args.query, mode=args.mode, reranker=args.reranker, top_k=args.top_k)
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r.score:.3f}] {r.doc_path}")
            print(f"   {r.heading_path}" if r.heading_path else "")
            snippet = r.snippet.replace("\n", " ")[:120]
            print(f"   {snippet}")
            print()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
