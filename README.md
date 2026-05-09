# litesearch

Pluggable semantic search over SQLite. Four search modes, four rerankers, time decay — all in a single `.db` file.

FTS5 for lexical search, [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector similarity. Bring any embedding backend.

---

## Install

```bash
git clone https://github.com/vedantvijay/litesearch.git
cd litesearch

# Option 1: install script (creates .venv automatically)
./install.sh              # everything
./install.sh core         # minimal — bring your own embedder
./install.sh ollama       # core + Ollama embedder
./install.sh server       # core + Ollama + REST API

# Option 2: pip
pip install -r requirements.txt        # all deps
pip install -r requirements-core.txt   # just the core
pip install -e .                       # editable install
```

### Requirements at a glance

| Package | Why | When needed |
|---------|-----|-------------|
| `sqlite-vec` | Vector columns + KNN queries in SQLite | Always |
| `python-frontmatter` | Parse YAML metadata from markdown files | Always |
| `pydantic` | LLM response parsing + API models | Always |
| `ollama` | Local embeddings via Ollama server | Default embedder |
| `fastapi` + `uvicorn` | REST API server | `litesearch serve` |
| `google-genai` | Gemini LLM reranker | `reranker="llm"` with Gemini backend |

---

## Quick start — Python library

```python
from litesearch import LiteSearch, OllamaEmbedder

# Connect to Ollama (must be running: ollama serve)
embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch("my.db", embedder=embedder)

# Index documents
engine.add("notes/ideas.md", "# Ideas\nBuild a search engine that just works.")
engine.add("notes/todo.md", "# TODO\n- Ship litesearch\n- Write docs")

# Search
results = engine.search("search engine", mode="hybrid", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.doc_path}: {r.snippet[:80]}")

# Clean up
engine.close()
```

### Index a directory of markdown files

```python
engine.add_directory("/path/to/obsidian-vault", glob="*.md")
```

### Index a single file from disk

```python
engine.add_file("/path/to/note.md", vault="/path/to")
```

### Context manager

```python
with LiteSearch("my.db", embedder=embedder) as engine:
    engine.add("doc.md", "content")
    results = engine.search("query")
```

---

## Search modes

| Mode | What it does |
|------|-------------|
| `bm25` | Full-text search via FTS5 with porter stemming. Weighted: body text (10x), doc title (8x), headings (5x), path (3x). |
| `semantic` | Vector similarity via sqlite-vec. L2 distance converted to cosine similarity. |
| `hybrid` | BM25 + semantic fused via reciprocal rank fusion (RRF, k=60). Best of both worlds. |
| `grep` | Substring / regex via ripgrep. Matches filenames first, then content. Falls back to SQL LIKE. |

```python
engine.search("deployment pipeline", mode="hybrid")
engine.search("kubernetes", mode="bm25")
engine.search("/error.*timeout/", mode="grep")  # regex with / prefix
```

---

## Rerankers

| Reranker | What it does |
|----------|-------------|
| `none` | Pass-through — first-stage scores only |
| `cross_encoder` | External CLI cross-encoder (e.g. BAAI/bge-reranker-v2-m3) |
| `llm` | LLM listwise reranking — Gemini 2.5 Flash (cloud) or Ollama (local) |
| `mmr` | Maximal marginal relevance — balances relevance vs diversity |
| `auto` | Runs all three, fuses via RRF |

```python
engine.search("query", reranker="cross_encoder")
engine.search("query", reranker="llm")       # needs gemini key or ollama
engine.search("query", reranker="mmr")       # diversity-aware
engine.search("query", reranker="auto")      # best quality, slowest
```

### LLM reranker setup

**Gemini (recommended — fast, ~1-2s):**
```python
from litesearch import LiteSearch, LiteSearchConfig
from litesearch.config import RerankerConfig

config = LiteSearchConfig(
    db_path="my.db",
    reranker=RerankerConfig(
        llm_judge_backend="gemini",
        gemini_api_key="your-key",
        # or: gemini_api_key_file="/path/to/key.txt"
    ),
)
engine = LiteSearch(config=config)
```

Or set `GEMINI_API_KEY` env var — it's picked up automatically.

**Ollama (local, private, slower):**
```python
config = LiteSearchConfig(
    db_path="my.db",
    reranker=RerankerConfig(
        llm_judge_backend="ollama",
        llm_judge_model="gemma4",
    ),
)
```

---

## Time decay

Blend relevance with recency. Recent documents score higher.

```python
results = engine.search(
    "deployment",
    time_decay=True,
    time_decay_half_life=90,   # days until score halves
    time_decay_weight=0.3,     # 0 = ignore recency, 1 = pure recency
)
```

---

## REST API

```bash
litesearch serve --db my.db --port 8900
```

### Endpoints

**Search:**
```
GET /search?q=kubernetes&mode=hybrid&reranker=none&top_k=10&group_by_doc=true
```

**Index a document:**
```
POST /index
{"path": "notes/new.md", "content": "# New Note\nContent here.", "title": "New Note"}
```

**Delete:**
```
DELETE /doc/notes/new.md
```

**Health check:**
```
GET /health
→ {"status": "ok", "documents": 142}
```

### Server options

```bash
litesearch serve \
  --db my.db \
  --host 0.0.0.0 \
  --port 8900 \
  --model embeddinggemma \
  --dims 768 \
  --ollama-url http://localhost:11434
```

---

## CLI

```bash
# Index a directory
litesearch index /path/to/notes --db my.db --glob "*.md"

# Search
litesearch query "kubernetes deployment" --db my.db --mode hybrid --top-k 5

# Start server
litesearch serve --db my.db
```

---

## Custom embedder

Implement three things: `dimensions`, `embed_documents`, `embed_query`.

```python
from litesearch import LiteSearch
from litesearch.embedder import Embedder

class OpenAIEmbedder:
    @property
    def dimensions(self) -> int:
        return 1536

    def embed_documents(self, texts):
        import openai
        resp = openai.embeddings.create(model="text-embedding-3-small", input=list(texts))
        return [e.embedding for e in resp.data]

    def embed_query(self, query):
        return self.embed_documents([query])[0]

engine = LiteSearch("my.db", embedder=OpenAIEmbedder())
```

Works with OpenAI, Cohere, Voyage, HuggingFace sentence-transformers, or anything else.

---

## How it works

```
Document → Parser → Chunker → Embedder → SQLite
              │          │          │
              │     heading-aware   │
              │     split at H2/H3  │
              │                     ▼
              │              sqlite-vec (vectors)
              │              FTS5 (full-text index)
              ▼
         frontmatter
         extraction
```

**Chunking:** Splits markdown at H2/H3 heading boundaries. Long sections get further split at paragraph boundaries (default soft max: 1500 chars). Each chunk carries a heading breadcrumb path (e.g. "Chapter 1 > Section 2").

**Indexing:** Hash-gated re-embedding — only re-embeds chunks whose content or title changed. Title is prepended to embedding text so semantic search covers filenames.

**Hybrid search:** Runs BM25 and semantic in parallel, fuses via reciprocal rank fusion (k=60). Each result gets `score = Σ 1/(60 + rank)` across both rankings.

**Group by doc:** Deduplicates results to best chunk per document (enabled by default).

---

## Architecture

```
litesearch/
├── __init__.py        LiteSearch class — the public API
├── __main__.py        CLI entry point
├── config.py          Dataclass-based configuration
├── db.py              SQLite + FTS5 + sqlite-vec schema
├── embedder.py        Embedder protocol + OllamaEmbedder
├── parser.py          Markdown frontmatter extraction
├── chunker.py         Heading-aware semantic chunking
├── indexer.py         Parse → chunk → embed → store pipeline
├── time_decay.py      Exponential recency weighting
├── server.py          Optional FastAPI REST layer
├── types.py           Candidate dataclass
├── search/
│   ├── bm25.py        FTS5 full-text search (weighted)
│   ├── semantic.py    Vector similarity search
│   ├── hybrid.py      BM25 + semantic RRF fusion
│   └── grep.py        Ripgrep / SQL LIKE fallback
└── rerank/
    ├── cross_encoder.py   External CLI cross-encoder
    ├── llm_judge.py       Gemini / Ollama listwise reranker
    ├── mmr.py             Maximal marginal relevance
    └── auto.py            All-reranker RRF fusion
```

---

## Configuration

```python
from litesearch import LiteSearch, LiteSearchConfig
from litesearch.config import (
    EmbeddingConfig, RerankerConfig, SearchConfig,
    TimeDecayConfig, ChunkingConfig, ServerConfig,
)

config = LiteSearchConfig(
    db_path="my.db",
    vault_path="/path/to/notes",           # for grep file search
    embedding=EmbeddingConfig(
        model="embeddinggemma",
        dimensions=768,
        ollama_url="http://localhost:11434",
    ),
    reranker=RerankerConfig(
        llm_judge_backend="gemini",
        gemini_api_key_file="~/.secrets/gemini_key",
        mmr_lambda=0.7,                    # relevance vs diversity
        rerank_cli="rerank",               # cross-encoder CLI path
    ),
    search=SearchConfig(
        default_top_k=10,
        candidate_pool_size=50,
    ),
    time_decay=TimeDecayConfig(
        default_half_life_days=90,
        default_weight=0.5,
    ),
    chunking=ChunkingConfig(
        soft_max_chars=1500,
    ),
    server=ServerConfig(
        host="0.0.0.0",
        port=8900,
        cors_origins=["*"],
    ),
)

engine = LiteSearch(config=config)
```

---

## License

MIT
