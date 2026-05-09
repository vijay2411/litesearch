"""Optional REST API server for litesearch.

Run with:  litesearch serve --db my.db --port 8900
Or:        python -m litesearch.server --db my.db
"""
from __future__ import annotations

import time
from typing import List, Literal, Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import LiteSearch, LiteSearchConfig
from .types import Candidate

Mode = Literal["grep", "bm25", "semantic", "hybrid"]
Reranker = Literal["none", "cross_encoder", "llm", "mmr", "auto"]


class SearchResult(BaseModel):
    chunk_id: int
    doc_id: int
    doc_path: str
    doc_title: str
    heading_path: str
    snippet: str
    text: str
    score: float
    doc_mtime: int
    start_line: int


class SearchMeta(BaseModel):
    mode: str
    reranker: str
    time_decay: bool
    took_ms: int
    pool_size: int
    total_candidates: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    meta: SearchMeta


class IndexRequest(BaseModel):
    path: str
    content: str
    title: Optional[str] = None


class IndexResponse(BaseModel):
    doc_id: int
    path: str


class DeleteResponse(BaseModel):
    deleted: bool
    path: str


def _to_result(c: Candidate) -> SearchResult:
    return SearchResult(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        doc_path=c.doc_path,
        doc_title=c.doc_title,
        heading_path=c.heading_path,
        snippet=c.snippet,
        text=c.text,
        score=float(c.score),
        doc_mtime=c.doc_mtime,
        start_line=c.start_line,
    )


def create_app(engine: LiteSearch) -> FastAPI:
    app = FastAPI(title="litesearch", version="0.1.0")
    app.state.engine = engine

    app.add_middleware(
        CORSMiddleware,
        allow_origins=engine.config.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/search", response_model=SearchResponse)
    def search(
        q: str = Query(..., min_length=1),
        mode: Mode = "hybrid",
        reranker: Optional[Reranker] = None,
        time_decay: bool = False,
        time_decay_half_life: Optional[int] = None,
        time_decay_weight: Optional[float] = None,
        top_k: Optional[int] = None,
        pool_size: Optional[int] = None,
        group_by_doc: bool = True,
    ):
        rk: Reranker = reranker or "none"
        t0 = time.perf_counter()

        results = engine.search(
            q,
            mode=mode,
            reranker=rk,
            top_k=top_k,
            pool_size=pool_size,
            group_by_doc=group_by_doc,
            time_decay=time_decay,
            time_decay_half_life=time_decay_half_life,
            time_decay_weight=time_decay_weight,
        )

        took_ms = int((time.perf_counter() - t0) * 1000)
        return SearchResponse(
            results=[_to_result(c) for c in results],
            meta=SearchMeta(
                mode=mode,
                reranker=rk,
                time_decay=time_decay,
                took_ms=took_ms,
                pool_size=pool_size or engine.config.search.candidate_pool_size,
                total_candidates=len(results),
            ),
        )

    @app.post("/index", response_model=IndexResponse)
    def index_doc(body: IndexRequest):
        doc_id = engine.add(body.path, body.content, title=body.title)
        return IndexResponse(doc_id=doc_id, path=body.path)

    @app.delete("/doc/{path:path}", response_model=DeleteResponse)
    def delete_doc(path: str):
        deleted = engine.remove(path)
        return DeleteResponse(deleted=deleted, path=path)

    @app.get("/health")
    def health():
        count = engine.conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        return {"status": "ok", "documents": count}

    return app


def run(
    db_path: str = "litesearch.db",
    host: str = "0.0.0.0",
    port: int = 8900,
    config: Optional[LiteSearchConfig] = None,
):
    import uvicorn

    from .config import ServerConfig

    cfg = config or LiteSearchConfig(
        db_path=db_path,
        server=ServerConfig(host=host, port=port),
    )
    engine = LiteSearch(config=cfg)
    app = create_app(engine)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)
