# 🔍 litesearch

**Semantic search that lives in a single SQLite file.**

Index any text. Search it four ways. Rerank with AI. No Elasticsearch, no Pinecone, no infra. Just a `.db` file you can `scp` to another machine.

---

## 🎯 What this is

A **local-first, file-based search engine** that combines full-text search (BM25) and vector similarity (embeddings) into one SQLite database. You feed it text — markdown, plain text, JSONL, whatever — and it gives you ranked, relevant results.

- 📦 **One file** — your entire search index is a single `.db` file. Copy it, back it up, version it.
- 🔀 **Four search modes** — keyword (BM25), meaning (semantic), both together (hybrid), or raw pattern matching (grep).
- 🧠 **Four rerankers** — cross-encoder, LLM judge, diversity-aware MMR, or all three fused together.
- ⏰ **Time decay** — blend relevance with recency so fresh documents surface first.
- 🔌 **Pluggable embeddings** — ships with Ollama support. Bring OpenAI, Cohere, HuggingFace, or anything else.
- 🌐 **REST API included** — run as a service, query from any language.

## ❌ What this is NOT

- **Not a hosted service** — there's no cloud, no API keys to litesearch itself. You run it.
- **Not a database replacement** — this is a search layer, not a primary data store.
- **Not production-scale** — built for thousands to low hundreds-of-thousands of documents. Not millions.
- **Not an embedding provider** — you need an embedding backend (Ollama, OpenAI, etc.). litesearch doesn't generate embeddings on its own.

## ✅ Why this exists

| Pain point | How litesearch solves it |
|---|---|
| "I need search but Elasticsearch is overkill" | SQLite. No server, no cluster, no ops. |
| "Pinecone/Weaviate need a hosted account" | Everything runs locally. Zero external dependencies at runtime. |
| "I want keyword AND semantic search" | Hybrid mode fuses both via reciprocal rank fusion. |
| "Setting up search infra takes days" | `pip install -e .` → 5 lines of Python → working search. |
| "I need to search my notes/docs/logs" | Feed it a directory. It handles chunking, embedding, indexing. |
| "My search results are stale" | Time decay scoring. Recent docs score higher. |

## 👤 Who this is for

- **Developers** building local tools, CLI apps, or agents that need search over text files
- **Note-takers** who want semantic search over Obsidian vaults, markdown notes, or plain text
- **AI/LLM builders** who need retrieval (RAG) without spinning up vector database infrastructure
- **Hobbyists & tinkerers** who want to understand how search engines work under the hood

## 🚫 Who this is NOT for

- **Teams needing multi-user concurrent writes** — SQLite has single-writer limitations
- **Anyone indexing millions of documents** — use a dedicated vector DB (Qdrant, Weaviate, etc.)
- **People who want a managed SaaS** — this is self-hosted, you maintain it
- **Production apps needing 99.9% uptime guarantees** — this is a library, not managed infra

---

## ⚙️ Technology under the hood

| Layer | Technology | Role |
|---|---|---|
| Storage | **SQLite** | Single-file relational database. Stores documents, chunks, metadata. |
| Full-text search | **FTS5** (built into SQLite) | BM25 keyword search with porter stemming. Weighted scoring across body, title, headings, path. |
| Vector search | **[sqlite-vec](https://github.com/asg017/sqlite-vec)** | SQLite extension that adds vector columns and KNN queries. Stores embeddings as FLOAT[] blobs. |
| Embeddings | **Ollama** (default) / any provider | Converts text → vectors. litesearch defines a protocol — plug in any backend. |
| REST API | **FastAPI** + **Uvicorn** | Optional HTTP layer for service mode. |
| Reranking | **Gemini** / **Ollama** / **CLI cross-encoder** | Optional second-pass scoring for better result quality. |

### 📋 System requirements

| Requirement | Why | How to check |
|---|---|---|
| **Python 3.11+** | Language runtime | `python3 --version` |
| **SQLite with extension loading** | sqlite-vec needs `enable_load_extension()` | See install notes below |
| **Ollama** (if using default embedder) | Generates embeddings locally | `ollama serve` then `ollama pull embeddinggemma` |

> ⚠️ **macOS users**: The default pyenv/system Python often ships without SQLite extension support. Use **Homebrew Python** (`brew install python@3.13`) — the install script detects this automatically.

---

## 🧩 How it works (plain English)

```
         ┌──────────────┐
         │  Your files   │  .md  .txt  .jsonl  .py  anything
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │    Parser     │  Detects format. Extracts frontmatter from markdown.
         └──────┬───────┘  Plain text? Just reads it. JSONL? One doc per line.
                │
         ┌──────▼───────┐
         │    Chunker    │  Splits long docs into pieces (~1500 chars).
         └──────┬───────┘  Markdown: splits at H2/H3 headings.
                │          Plain text: splits at paragraph boundaries.
                │          Each chunk remembers its heading breadcrumb path.
                │
         ┌──────▼───────┐
         │   Embedder    │  Converts each chunk into a vector (list of numbers).
         └──────┬───────┘  A title like "Deployment Guide" becomes [0.12, -0.34, ...]
                │          Similar meaning → similar vectors → findable by search.
                │
         ┌──────▼───────┐
         │    SQLite     │  Stores everything in ONE file:
         │               │
         │  ┌──────────┐ │  documents table  → full text, metadata, timestamps
         │  │ FTS5     │ │  chunks_fts       → keyword index (BM25 scoring)
         │  │ sqlite-  │ │  chunks_vec       → vector index (similarity scoring)
         │  │ vec      │ │
         │  └──────────┘ │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │   Search      │  Query hits BOTH indexes, fuses results:
         │               │
         │  BM25:    "exact keyword matches, stemmed"
         │  Semantic: "similar meaning, even different words"
         │  Hybrid:  "both combined via rank fusion"
         │  Grep:    "raw pattern matching via ripgrep"
         │               │
         └──────┬───────┘
                │
         ┌──────▼───────┐
         │   Reranker    │  Optional second pass. Re-scores top results using:
         │  (optional)   │  • Cross-encoder (ML model, most accurate)
         └──────┬───────┘  • LLM judge (Gemini/Ollama rates relevance)
                │          • MMR (picks diverse results, reduces redundancy)
                │          • Auto (all three, fused together)
                │
         ┌──────▼───────┐
         │   Results     │  Ranked list. Each result has:
         │               │  score, doc_path, snippet, heading, line numbers
         └──────────────┘
```

### 🔑 Key concepts

**Hash-gated re-embedding** — Each chunk gets a SHA-256 hash of its title + text. If you re-index a file and the content hasn't changed, the embedding is reused. Only changed chunks get re-embedded. This makes re-indexing fast.

**Reciprocal Rank Fusion (RRF)** — When combining BM25 and semantic results, each result gets `score = Σ 1/(60 + rank)` across both rankings. This normalizes scores from completely different systems into one unified ranking.

**Group by document** — By default, results are deduplicated to show only the best-matching chunk per document. You get 10 different documents, not 10 chunks from the same doc.

**Title-aware embeddings** — The document title is prepended to each chunk before embedding: `"Deployment Guide | chunk text here"`. This means searching for "deployment" can find the right document even if the chunk itself doesn't contain that word.

---

## 🚀 Quick start

```bash
git clone https://github.com/vijay2411/search-md-files.git
cd search-md-files
./install.sh        # creates .venv, installs everything, verifies sqlite-vec
source .venv/bin/activate
```

```python
from litesearch import LiteSearch, OllamaEmbedder

embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch("my.db", embedder=embedder)

engine.add("notes/ideas.md", "# Ideas\nBuild a search engine that just works.")
engine.add("notes/todo.md", "# TODO\n- Ship litesearch\n- Write docs")

results = engine.search("search engine", mode="hybrid", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.doc_path}: {r.snippet[:80]}")

engine.close()
```

> 📖 **Full usage guide** — search modes, rerankers, time decay, REST API, CLI, custom embedders, configuration, and service integration — see **[USAGE.md](USAGE.md)**.

---

## 📁 Supported formats

| Format | Extensions | Behavior |
|---|---|---|
| 📝 Markdown | `.md` `.markdown` `.mdx` `.mdown` | YAML frontmatter extracted, H1 → title, heading-aware chunking |
| 📄 Plain text | `.txt` `.py` `.js` `.csv` — anything else | Filename → title, body indexed as-is |
| 📋 JSONL | `.jsonl` | One JSON object per line → one document per line |

---

## 🏗️ Project structure

```
litesearch/
├── __init__.py        LiteSearch class — the public API
├── __main__.py        CLI entry point
├── config.py          Dataclass-based configuration
├── db.py              SQLite + FTS5 + sqlite-vec schema
├── embedder.py        Embedder protocol + OllamaEmbedder
├── parser.py          Format-agnostic file parsing
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

## 📄 License

MIT
