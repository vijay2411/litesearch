"""litesearch — pluggable semantic search over SQLite.

Usage as a library:

    from litesearch import LiteSearch, OllamaEmbedder

    embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
    engine = LiteSearch("my.db", embedder=embedder)

    engine.add("notes/hello.md", "# Hello\\nSome content here.")
    results = engine.search("hello", mode="hybrid")

Usage as a REST API:

    litesearch serve --db my.db --port 8900
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Literal, Optional

from .config import LiteSearchConfig
from .db import connect, init_db
from .embedder import Embedder, OllamaEmbedder, from_blob, normalize, to_blob
from .indexer import bootstrap_directory, index_file, index_jsonl, index_text, remove_doc
from .rerank.auto import rerank_auto
from .rerank.cross_encoder import rerank_cross_encoder
from .rerank.llm_judge import rerank_llm
from .rerank.mmr import rerank_mmr
from .search.bm25 import bm25_search
from .search.grep import grep_search
from .search.hybrid import hybrid_search
from .search.semantic import query_vector, semantic_search
from .time_decay import apply_time_decay
from .types import Candidate

__version__ = "0.1.0"
__all__ = [
    "LiteSearch",
    "LiteSearchConfig",
    "Candidate",
    "Embedder",
    "OllamaEmbedder",
]

Mode = Literal["grep", "bm25", "semantic", "hybrid"]
Reranker = Literal["none", "cross_encoder", "llm", "mmr", "auto"]


class LiteSearch:
    def __init__(
        self,
        db_path: str | Path = "litesearch.db",
        *,
        embedder: Optional[Embedder] = None,
        config: Optional[LiteSearchConfig] = None,
    ):
        self.config = config or LiteSearchConfig(db_path=str(db_path))
        self._embedder = embedder or OllamaEmbedder(
            model=self.config.embedding.model,
            ollama_url=self.config.embedding.ollama_url,
            dims=self.config.embedding.dimensions,
        )
        self._conn = connect(self.config.db_path, dim=self._embedder.dimensions)
        init_db(self._conn, dim=self._embedder.dimensions)

    def _resolve_gemini_key(self) -> str:
        key = self.config.reranker.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key and self.config.reranker.gemini_api_key_file:
            try:
                key = Path(self.config.reranker.gemini_api_key_file).read_text().strip()
            except OSError:
                pass
        return key

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def add(self, path: str, content: str, *, title: Optional[str] = None) -> int:
        """Index a document from raw text. Returns the document id."""
        return index_text(
            self._conn, path, content, self._embedder,
            title=title, soft_max_chars=self.config.chunking.soft_max_chars,
        )

    def add_file(self, file_path: str | Path, *, vault: Optional[str | Path] = None) -> bool:
        """Index a file from disk. Returns True if content changed."""
        p = Path(file_path)
        v = Path(vault) if vault else None
        return index_file(
            self._conn, p, self._embedder,
            vault=v, soft_max_chars=self.config.chunking.soft_max_chars,
        )

    def add_jsonl(
        self,
        file_path: str | Path,
        *,
        text_field: str = "text",
        title_field: str = "title",
    ) -> int:
        """Index every line of a JSONL file as a separate document. Returns count indexed."""
        return index_jsonl(
            self._conn, Path(file_path), self._embedder,
            text_field=text_field, title_field=title_field,
            soft_max_chars=self.config.chunking.soft_max_chars,
        )

    def add_directory(
        self, directory: str | Path, *, glob: str = "*.*"
    ) -> int:
        """Index all matching files in a directory. Returns count of files indexed."""
        return bootstrap_directory(
            self._conn, Path(directory), self._embedder,
            soft_max_chars=self.config.chunking.soft_max_chars, glob=glob,
        )

    def remove(self, path: str) -> bool:
        """Remove a document by path. Returns True if found and removed."""
        return remove_doc(self._conn, path)

    def search(
        self,
        query: str,
        *,
        mode: Mode = "hybrid",
        reranker: Reranker = "none",
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
        group_by_doc: bool = True,
        time_decay: bool = False,
        time_decay_half_life: Optional[int] = None,
        time_decay_weight: Optional[float] = None,
    ) -> List[Candidate]:
        k = top_k or self.config.search.default_top_k
        pool = pool_size or self.config.search.candidate_pool_size
        needs_vec = reranker in ("mmr", "auto")

        if mode == "grep":
            candidates = grep_search(self._conn, query, top_k=pool, vault_path=self.config.vault_path)
        elif mode == "bm25":
            candidates = bm25_search(self._conn, query, top_k=pool)
        elif mode == "semantic":
            candidates = semantic_search(self._conn, query, pool, self._embedder, keep_vectors=needs_vec)
        elif mode == "hybrid":
            candidates = hybrid_search(self._conn, query, pool, self._embedder, keep_vectors=needs_vec)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

        if group_by_doc:
            seen_docs: set[int] = set()
            deduped: list[Candidate] = []
            for c in candidates:
                if c.doc_id in seen_docs:
                    continue
                seen_docs.add(c.doc_id)
                deduped.append(c)
            candidates = deduped

        if reranker == "none" or not candidates:
            ranked = candidates[:k]
        elif reranker == "cross_encoder":
            ranked = rerank_cross_encoder(query, candidates, k, cli=self.config.reranker.rerank_cli)
        elif reranker == "mmr":
            qvec = query_vector(query, self._embedder)
            ranked = rerank_mmr(
                self._conn, qvec, candidates, k,
                dim=self._embedder.dimensions, lambda_=self.config.reranker.mmr_lambda,
            )
        elif reranker == "llm":
            ranked = rerank_llm(
                query, candidates, k,
                backend=self.config.reranker.llm_judge_backend,
                gemini_api_key=self._resolve_gemini_key(),
                gemini_model=self.config.reranker.gemini_model,
                ollama_url=self.config.embedding.ollama_url,
                ollama_model=self.config.reranker.llm_judge_model,
            )
        elif reranker == "auto":
            qvec = query_vector(query, self._embedder)
            ranked = rerank_auto(
                self._conn, query, qvec, candidates, k,
                dim=self._embedder.dimensions,
                mmr_lambda=self.config.reranker.mmr_lambda,
                rerank_cli=self.config.reranker.rerank_cli,
                llm_backend=self.config.reranker.llm_judge_backend,
                gemini_api_key=self._resolve_gemini_key(),
                gemini_model=self.config.reranker.gemini_model,
                ollama_url=self.config.embedding.ollama_url,
                ollama_model=self.config.reranker.llm_judge_model,
            )
        else:
            raise ValueError(f"Unknown reranker: {reranker}")

        if time_decay:
            hl = time_decay_half_life or self.config.time_decay.default_half_life_days
            w = time_decay_weight if time_decay_weight is not None else self.config.time_decay.default_weight
            ranked = apply_time_decay(list(ranked), half_life_days=hl, weight=w)

        return ranked

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
