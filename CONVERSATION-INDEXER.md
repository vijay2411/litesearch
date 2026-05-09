# 🔌 Plugging litesearch into Claude Code conversation search

How to upgrade a basic substring-matching `/search-conversations` skill into semantic search using litesearch as the backend.

---

## 🧩 Architecture

```
┌──────────────────────────────────────────┐
│  Cron / launchd (every 5–10 min)         │
│                                          │
│  runs: conversation-indexer.py           │
│    1. Scans ~/.claude/projects/*/*.jsonl  │
│    2. Parses each JSONL → extracts text   │
│    3. Groups turns into chunks            │
│    4. Calls engine.add() per chunk        │
│    5. Hash-gated — skips unchanged files  │
│                                          │
│  Writes to: ~/.claude/conversations.db   │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  Skill: /search-conversations            │
│                                          │
│  runs: search.py "your query"            │
│    1. Opens ~/.claude/conversations.db   │
│    2. engine.search(query, mode="hybrid")│
│    3. Returns ranked results with:       │
│       - session ID (for /resume)         │
│       - project name                     │
│       - timestamp                        │
│       - matching snippet                 │
│                                          │
│  No server needed. Just reads SQLite.    │
└──────────────────────────────────────────┘
```

**Key insight:** No daemon or server required. The indexer writes to a `.db` file on a schedule. The skill reads from it on demand. SQLite handles this cleanly — they never run simultaneously, and even if they did, SQLite supports concurrent readers.

---

## 📄 Claude Code conversation format

Conversations live at `~/.claude/projects/<project-dir>/<session-uuid>.jsonl`.

Each line is a JSON object:

```jsonl
{"type": "user",      "message": {"role": "user",      "content": "how do I deploy this?"}, "sessionId": "abc-123", "timestamp": "2026-05-09T10:30:00Z", ...}
{"type": "assistant", "message": {"role": "assistant",  "content": [{"type": "text", "text": "You can deploy by..."}]}, "sessionId": "abc-123", "timestamp": "2026-05-09T10:30:05Z", ...}
{"type": "assistant", "message": {"role": "assistant",  "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "git push"}}]}, ...}
```

**What to extract:**
- `type: "user"` → `message.content` (always a string)
- `type: "assistant"` → `message.content` is either a string or a list of blocks:
  - `{"type": "text", "text": "..."}` → the actual response text
  - `{"type": "tool_use", ...}` → optionally include command names, skip verbose output
  - `{"type": "tool_result", ...}` → skip (too noisy, mostly file contents)

**What to ignore:**
- `type: "attachment"` lines (hook outputs, system context)
- Lines where `content` starts with `<local-command-` or `<command-name>` (CLI meta-messages)
- Tool results (they're huge and mostly file dumps)

**Metadata to preserve per chunk:**
- `sessionId` — needed for `/resume <uuid>`
- `timestamp` — for time decay and display
- Project directory name — tells you which repo the conversation was about

---

## 🔨 Indexer script outline

The indexer (`conversation-indexer.py`) needs to:

### 1. Discover conversation files

```python
import glob
files = glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))
```

### 2. Skip already-indexed files (by file hash or mtime)

litesearch already does hash-gated re-embedding at the chunk level. But you can skip entire files early by checking mtime:

```python
import os
mtime = os.path.getmtime(path)
# Compare against last-indexed mtime stored in a simple JSON sidecar
# or just let litesearch's content_hash handle dedup — it's fast enough
```

### 3. Parse each file into a "document"

Don't index every JSONL line as a separate document — that's too granular. Instead, **group consecutive turns into conversation chunks** of ~1500 chars:

```python
def parse_conversation(jsonl_path):
    """Yield (text, metadata) tuples — one per conversation chunk."""
    entries = [json.loads(line) for line in open(jsonl_path)]

    session_id = None
    project = os.path.basename(os.path.dirname(jsonl_path))
    buffer = []
    buffer_chars = 0

    for entry in entries:
        if entry.get("type") not in ("user", "assistant"):
            continue

        text = extract_text(entry)  # reuse your existing extract_text()
        if not text or text.startswith("<"):
            continue

        session_id = entry.get("sessionId", session_id)
        timestamp = entry.get("timestamp", "")
        role = entry.get("type", "?")

        line = f"[{role}] {text}"
        buffer.append(line)
        buffer_chars += len(line)

        # Flush when buffer exceeds ~1500 chars
        if buffer_chars >= 1500:
            yield {
                "text": "\n".join(buffer),
                "session_id": session_id,
                "project": project,
                "timestamp": timestamp,
                "path": jsonl_path,
            }
            buffer = []
            buffer_chars = 0

    # Flush remaining
    if buffer:
        yield {
            "text": "\n".join(buffer),
            "session_id": session_id,
            "project": project,
            "timestamp": timestamp,
            "path": jsonl_path,
        }
```

### 4. Feed into litesearch

```python
from litesearch import LiteSearch, OllamaEmbedder

DB_PATH = os.path.expanduser("~/.claude/conversations.db")

embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch(DB_PATH, embedder=embedder)

for jsonl_path in files:
    for chunk in parse_conversation(jsonl_path):
        # Use a path that encodes session + chunk index for uniqueness
        doc_path = f"{chunk['project']}/{chunk['session_id']}"
        engine.add(
            path=doc_path,
            content=chunk["text"],
            title=f"{chunk['project']} — {chunk['timestamp'][:10]}",
        )

engine.close()
```

**Note:** Each conversation chunk becomes a litesearch document. The `doc_path` includes session ID so the search skill can extract it for `/resume`.

---

## 🔎 Search script outline

The search script (`search.py`) becomes much simpler:

```python
from litesearch import LiteSearch, OllamaEmbedder

DB_PATH = os.path.expanduser("~/.claude/conversations.db")

embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch(DB_PATH, embedder=embedder)

results = engine.search(query, mode="hybrid", top_k=10)

for r in results:
    # doc_path is "project-name/session-uuid"
    project, session_id = r.doc_path.rsplit("/", 1)
    print(f"── {r.doc_title} | session {session_id[:8]} | {project}")
    print(f"   {r.snippet[:200]}")
    print(f"   → /resume {session_id}")
    print()

engine.close()
```

**What changes from the current skill:**
- `mode="hybrid"` means "deployment pipeline" finds conversations about "CI/CD setup"
- Time decay means recent conversations rank higher
- Results are ranked by relevance, not just "newest match first"
- `/resume` still works — session UUID is encoded in the doc_path

---

## ⏰ Scheduling the indexer

### macOS (launchd)

Create `~/Library/LaunchAgents/com.litesearch.conversations.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.litesearch.conversations</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/litesearch/.venv/bin/python</string>
        <string>/path/to/conversation-indexer.py</string>
    </array>
    <key>StartInterval</key>
    <integer>300</integer>
    <key>StandardOutPath</key>
    <string>/tmp/litesearch-indexer.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/litesearch-indexer.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.litesearch.conversations.plist
```

### Linux (cron)

```bash
crontab -e
# Add:
*/5 * * * * /path/to/litesearch/.venv/bin/python /path/to/conversation-indexer.py >> /tmp/litesearch-indexer.log 2>&1
```

### Manual

```bash
# Just run it whenever
python conversation-indexer.py
```

---

## ⚠️ Gotchas

1. **Ollama must be running** when the indexer runs. If it's not, embedding will fail silently. The indexer should check and skip gracefully.

2. **First run is slow.** 82 conversation files × multiple chunks each × embedding calls. Expect a few minutes. Subsequent runs are fast — hash-gated, only new/changed files get embedded.

3. **Don't index tool results.** They're huge (full file contents, command outputs) and add noise. Only index `[user]` and `[assistant]` text blocks.

4. **Session ID uniqueness.** Multiple chunks from the same conversation share a session ID. The search skill should deduplicate results per session (litesearch's `group_by_doc=True` handles this if each session maps to one doc, or you handle it in the skill).

5. **Database location.** `~/.claude/conversations.db` keeps it alongside the source data. Don't put it inside a git repo.

6. **Stale index fallback.** If the `.db` doesn't exist or litesearch isn't installed, the skill should fall back to the current substring search. Don't break the skill for people who haven't set up indexing.
