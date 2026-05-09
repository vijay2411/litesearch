# 📖 litesearch — Full Usage Guide

Everything you need to know about using litesearch: install, library API, search modes, rerankers, time decay, REST API, CLI, custom embedders, configuration, and service integration.

---

## 📦 Install

```bash
git clone https://github.com/vijay2411/search-md-files.git
cd search-md-files

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

### 📋 Dependencies

| Package | Why | When needed |
|---|---|---|
| `sqlite-vec` | Vector columns + KNN queries in SQLite | Always |
| `python-frontmatter` | Parse YAML metadata from markdown files | Always |
| `pydantic` | LLM response parsing + API models | Always |
| `ollama` | Local embeddings via Ollama server | Default embedder |
| `fastapi` + `uvicorn` | REST API server | `litesearch serve` |
| `google-genai` | Gemini LLM reranker | `reranker="llm"` with Gemini backend |

---

## 🐍 Python library

### Basic usage

```python
from litesearch import LiteSearch, OllamaEmbedder

embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch("my.db", embedder=embedder)

# Index documents
engine.add("notes/ideas.md", "# Ideas\nBuild a search engine that just works.")
engine.add("notes/todo.md", "# TODO\n- Ship litesearch\n- Write docs")

# Search
results = engine.search("search engine", mode="hybrid", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.doc_path}: {r.snippet[:80]}")

engine.close()
```

### Index a directory

```python
engine.add_directory("/path/to/obsidian-vault", glob="*.md")   # markdown only
engine.add_directory("/path/to/docs")                           # all files (default: *.*)
engine.add_directory("/path/to/logs", glob="*.txt")             # plain text
```

### Index a single file

```python
engine.add_file("/path/to/note.md", vault="/path/to")    # markdown with frontmatter
engine.add_file("/path/to/readme.txt")                    # plain text
```

### Index a JSONL file

Each line becomes a separate document. Configure which JSON keys hold the text and title.

```python
engine.add_jsonl("conversations.jsonl", text_field="text", title_field="title")
```

### Context manager

```python
with LiteSearch("my.db", embedder=embedder) as engine:
    engine.add("doc.md", "content")
    results = engine.search("query")
```

---

## 🔎 Search modes

| Mode | What it does |
|---|---|
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

## 🏆 Rerankers

| Reranker | What it does |
|---|---|
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

## ⏰ Time decay

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

## 🌐 REST API

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

**Batch index:**
```
POST /index/batch
{"documents": [
  {"path": "a.md", "content": "# A\nFirst doc"},
  {"path": "b.txt", "content": "Second doc", "title": "B"}
]}
→ {"indexed": 2, "doc_ids": [...]}
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

## 💻 CLI

```bash
# Index a directory (all files by default)
litesearch index /path/to/notes --db my.db
litesearch index /path/to/notes --db my.db --glob "*.md"    # markdown only
litesearch index /path/to/logs --db my.db --glob "*.txt"    # plain text

# Index a JSONL file
litesearch index-jsonl conversations.jsonl --db my.db --text-field text --title-field title

# Search
litesearch query "kubernetes deployment" --db my.db --mode hybrid --top-k 5

# Start server
litesearch serve --db my.db
```

---

## 🔌 Custom embedder

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

## ⚙️ Configuration

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

## 🚀 Running as a service

litesearch runs as a persistent HTTP service. Start it once, query it from any tool or script.

```bash
# Start in background
litesearch serve --db my.db --port 8900 &

# Or with nohup for persistence
nohup litesearch serve --db my.db --port 8900 > litesearch.log 2>&1 &

# Or with systemd (Linux)
# Create /etc/systemd/system/litesearch.service, then:
# systemctl enable --now litesearch
```

### Integration with Claude Code skills

A skill can query litesearch via simple HTTP calls:

```bash
# Search
curl -s "http://localhost:8900/search?q=deployment+pipeline&mode=hybrid&top_k=5" \
  | jq '.results[] | {path: .doc_path, score, snippet}'

# Index a document
curl -s -X POST http://localhost:8900/index \
  -H "Content-Type: application/json" \
  -d '{"path": "conv/2024-01-15.md", "content": "# Meeting Notes\n...", "title": "Meeting Notes"}'

# Batch index
curl -s -X POST http://localhost:8900/index/batch \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"path": "a.md", "content": "..."}, {"path": "b.md", "content": "..."}]}'
```

The REST API returns JSON — any language or tool that makes HTTP requests can use it. No SDK needed on the client side.
