# 🔌 Plugging litesearch into Claude Code conversation search

Upgrade a substring-matching `/search-conversations` skill into semantic search using litesearch. Find conversations by meaning, not just keywords.

---

## 🧩 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Cron / launchd (every 5–10 min)                            │
│                                                             │
│  runs: conversation-indexer.py                              │
│    1. Scans ~/.claude/projects/*/*.jsonl                    │
│    2. Parses each session → extracts user/assistant text    │
│    3. Groups turns into ~1500-char chunks                   │
│    4. Feeds chunks into the "conversations" index           │
│    5. Hash-gated — skips files that haven't changed         │
│                                                             │
│  Writes to: ~/.litesearch/conversations.db                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Skill: /search-conversations                               │
│                                                             │
│  runs: litesearch search conversations "your query"         │
│    → returns ranked results with:                           │
│       - session ID (for /resume)                            │
│       - project name                                        │
│       - timestamp                                           │
│       - matching snippet                                    │
│                                                             │
│  No server needed. Just reads the SQLite file directly.     │
└─────────────────────────────────────────────────────────────┘
```

**No daemon, no server.** The indexer writes to a named index on a schedule. The skill reads from it on demand via the `litesearch search` CLI. SQLite handles concurrent readers cleanly.

---

## 📄 Claude Code conversation format

Conversations live at `~/.claude/projects/<project-dir>/<session-uuid>.jsonl`.

Each line is a JSON object:

```jsonl
{"type": "user",      "message": {"role": "user",      "content": "how do I deploy this?"}, "sessionId": "abc-123", "timestamp": "2026-05-09T10:30:00Z", ...}
{"type": "assistant", "message": {"role": "assistant",  "content": [{"type": "text", "text": "You can deploy by..."}]}, "sessionId": "abc-123", "timestamp": "2026-05-09T10:30:05Z", ...}
{"type": "assistant", "message": {"role": "assistant",  "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "git push"}}]}, ...}
```

### ✅ What to extract

| Source | Extract |
|---|---|
| `type: "user"` | `message.content` (always a string) |
| `type: "assistant"` with `"type": "text"` blocks | The `text` field from each block |
| `type: "assistant"` with `"type": "tool_use"` blocks | Optionally the tool name + command (for context) |

### ❌ What to skip

| Source | Why |
|---|---|
| `type: "attachment"` | Hook outputs, system context — noise |
| `content` starting with `<local-command-` or `<command-name>` | CLI meta-messages |
| `"type": "tool_result"` blocks | Huge file dumps, command outputs — drowns out signal |

### 📎 Metadata to preserve per chunk

| Field | Why |
|---|---|
| `sessionId` | Needed for `/resume <uuid>` |
| `timestamp` | Time decay scoring + display |
| Project directory name | Which repo the conversation was about |

---

## 🔨 Building the indexer

Create `~/.claude/skills/search-conversations/conversation-indexer.py`:

### Step 1: Extract text from conversation entries

Reuse the `extract_text()` function from the existing skill, but filter out noise:

```python
def extract_text(entry):
    """Pull readable text from a user or assistant message."""
    if entry.get("type") not in ("user", "assistant"):
        return ""
    message = entry.get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        if content.startswith("<"):  # skip CLI meta-messages
            return ""
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                name = block.get("name", "")
                cmd = (block.get("input") or {}).get("command", "")
                if cmd:
                    parts.append(f"[{name}] {cmd}")
            # skip tool_result — too noisy
        return "\n".join(parts)

    return ""
```

### Step 2: Group turns into chunks

Don't index every line separately — group consecutive turns into ~1500-char chunks:

```python
import json, os, glob

def parse_conversation(jsonl_path):
    """Yield chunks from a conversation file."""
    session_id = None
    project = os.path.basename(os.path.dirname(jsonl_path))
    buffer = []
    buffer_chars = 0
    chunk_idx = 0

    with open(jsonl_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = extract_text(entry)
            if not text:
                continue

            session_id = entry.get("sessionId", session_id)
            timestamp = entry.get("timestamp", "")
            role = entry.get("type", "?")

            turn = f"[{role}] {text}"
            buffer.append(turn)
            buffer_chars += len(turn)

            if buffer_chars >= 1500:
                yield {
                    "text": "\n".join(buffer),
                    "session_id": session_id,
                    "project": project,
                    "timestamp": timestamp,
                    "chunk_idx": chunk_idx,
                }
                buffer = []
                buffer_chars = 0
                chunk_idx += 1

    if buffer:
        yield {
            "text": "\n".join(buffer),
            "session_id": session_id,
            "project": project,
            "timestamp": timestamp,
            "chunk_idx": chunk_idx,
        }
```

### Step 3: Feed into litesearch

```python
from litesearch import LiteSearch, OllamaEmbedder

INDEX_NAME = "conversations"
PROJECTS_DIR = os.path.expanduser("~/.claude/projects")

embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
engine = LiteSearch(
    os.path.expanduser(f"~/.litesearch/{INDEX_NAME}.db"),
    embedder=embedder,
)

files = sorted(glob.glob(os.path.join(PROJECTS_DIR, "*/*.jsonl")))
indexed = 0

for jsonl_path in files:
    for chunk in parse_conversation(jsonl_path):
        doc_path = f"{chunk['project']}/{chunk['session_id']}#{chunk['chunk_idx']}"
        engine.add(
            path=doc_path,
            content=chunk["text"],
            title=f"{chunk['project']} — {chunk['timestamp'][:10]}",
        )
        indexed += 1

engine.close()
print(f"Indexed {indexed} chunks from {len(files)} conversation files")
```

**How `doc_path` works:** It encodes `project/session-uuid#chunk-index` so the search skill can:
- Extract the session UUID for `/resume`
- Show which project the conversation was in
- Deduplicate results per session with `group_by_doc`

### Step 4: Put it all together

The full script is roughly 80 lines. Combine steps 1-3 into a single file with a `__main__` guard:

```python
if __name__ == "__main__":
    # ... the code from step 3 above ...
```

---

## 🔎 Updating the search skill

### Option A: Use the litesearch CLI directly (simplest)

Update `SKILL.md` to run litesearch search instead of the old `search.py`:

```markdown
## How to run

\```bash
litesearch search conversations "your query" --top-k 10
\```
```

The output already includes doc paths (which encode session IDs) and snippets. The skill just needs to parse the output and offer `/resume`.

### Option B: Keep a Python search script (more control)

Replace `search.py` with a version that uses litesearch:

```python
#!/usr/bin/env python3
"""Search past conversations using litesearch semantic search."""

import argparse
import os
import sys

from litesearch import LiteSearch, OllamaEmbedder

INDEX_PATH = os.path.expanduser("~/.litesearch/conversations.db")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", help="Search query")
    p.add_argument("-n", "--max", type=int, default=10)
    p.add_argument("--mode", default="semantic",
                   choices=["semantic", "hybrid", "bm25", "grep"])
    args = p.parse_args()

    if not os.path.exists(INDEX_PATH):
        print("No conversation index found. Run conversation-indexer.py first.")
        sys.exit(1)

    embedder = OllamaEmbedder(model="embeddinggemma", dims=768)
    engine = LiteSearch(INDEX_PATH, embedder=embedder)

    results = engine.search(
        args.query,
        mode=args.mode,
        top_k=args.max,
        group_by_doc=True,
    )

    if not results:
        print(f"No matches for '{args.query}'.")
        engine.close()
        return

    print(f"Found {len(results)} match(es) for '{args.query}':\n")

    for r in results:
        # doc_path format: "project-dir/session-uuid#chunk-index"
        path_part = r.doc_path.split("#")[0]  # strip chunk index
        if "/" in path_part:
            project, session_id = path_part.rsplit("/", 1)
        else:
            project, session_id = "unknown", path_part

        print(f"── [{r.score:.3f}] {r.doc_title} | session {session_id[:8]} | {project}")
        snippet = r.snippet.replace("\n", " ")[:200]
        print(f"   {snippet}")
        print(f"   → /resume {session_id}")
        print()

    engine.close()


if __name__ == "__main__":
    main()
```

### Option C: Fallback to substring search

For robustness, check if the litesearch index exists. If not, fall back to the existing substring search:

```python
INDEX_PATH = os.path.expanduser("~/.litesearch/conversations.db")

if os.path.exists(INDEX_PATH):
    # Use litesearch (semantic search)
    search_with_litesearch(query, args)
else:
    # Fall back to old substring search
    search_with_substring(query, args)
```

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
        <string>/Users/you/.claude/skills/search-conversations/conversation-indexer.py</string>
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
*/5 * * * * /path/to/litesearch/.venv/bin/python ~/.claude/skills/search-conversations/conversation-indexer.py >> /tmp/litesearch-indexer.log 2>&1
```

### Manual (one-shot)

```bash
python ~/.claude/skills/search-conversations/conversation-indexer.py
```

Or just use the CLI directly:

```bash
# Quick test — search immediately after indexing
python conversation-indexer.py && litesearch search conversations "deployment pipeline"
```

---

## 🔄 What changes from the old skill

| Before (substring) | After (litesearch) |
|---|---|
| `q not in text.lower()` — exact keyword match only | Semantic search — "deployment pipeline" finds "CI/CD setup" |
| Scans every JSONL file on every search | Pre-indexed, instant lookup |
| Sorted by timestamp only | Ranked by relevance (+ optional time decay) |
| Slow with many conversations | Fast — SQLite indexed queries |
| No understanding of meaning | Embeddings capture semantic similarity |
| `/resume` works via session ID in output | `/resume` still works — session UUID encoded in `doc_path` |

---

## ⚠️ Gotchas

1. **Ollama must be running** when the indexer runs. The indexer should check and skip gracefully if it can't connect.

2. **First run is slow.** Many conversation files × multiple chunks × embedding calls. Expect a few minutes. Subsequent runs are fast — only new/changed files get processed.

3. **Don't index tool results.** They're huge (full file contents, command outputs) and add noise. Only index user and assistant text.

4. **Session deduplication.** Multiple chunks from the same conversation share a session ID. Use `group_by_doc=True` (default) so results show one hit per session, not per chunk.

5. **Database location.** The named index lives at `~/.litesearch/conversations.db`. Don't put it inside a git repo. Run `litesearch info conversations` to inspect it.

6. **Stale index fallback.** If `~/.litesearch/conversations.db` doesn't exist, the skill should fall back to the old substring search. Don't break the skill for people who haven't set up indexing yet.

7. **Verifying it works.** After running the indexer:
   ```bash
   litesearch info conversations         # check doc/chunk counts
   litesearch search conversations "test query"  # verify results
   ```
