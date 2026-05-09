from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Optional

from ..types import Candidate

log = logging.getLogger("litesearch.grep")


def _make_snippet(text: str, query: str, max_chars: int = 240) -> str:
    lower = text.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx < 0:
        return text[:max_chars]
    half = max_chars // 2
    start = max(0, idx - half)
    end = min(len(text), start + max_chars)
    snippet = text[start:end]
    out = snippet.replace(text[idx : idx + len(query)], f"<b>{text[idx:idx+len(query)]}</b>")
    return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")


def _resolve_chunk(conn: sqlite3.Connection, doc_id: int, line_no: int):
    return conn.execute(
        "SELECT id, heading_path, text, start_line FROM chunks "
        "WHERE doc_id = ? AND start_line <= ? AND end_line >= ? "
        "ORDER BY chunk_index LIMIT 1",
        (doc_id, line_no, line_no),
    ).fetchone()


def grep_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int,
    vault_path: Optional[str | Path] = None,
) -> List[Candidate]:
    if not query:
        return []

    is_regex = query.startswith("/")
    pattern = query[1:] if is_regex else query
    if not pattern:
        return []

    vault = Path(vault_path) if vault_path else None
    rg = shutil.which("rg")
    matches: list[tuple[str, int, str]] = []

    if rg and vault and vault.exists():
        # First: match filenames against the pattern (case-insensitive).
        if not is_regex:
            try:
                fname_proc = subprocess.run(
                    [rg, "--files", "--glob", "*.md", str(vault)],
                    capture_output=True, text=True, timeout=10,
                )
                pat_lower = pattern.lower()
                for fline in fname_proc.stdout.splitlines():
                    p = Path(fline.strip())
                    try:
                        rel = str(p.relative_to(vault))
                    except ValueError:
                        rel = str(p)
                    if not rel.endswith(".md"):
                        continue
                    if pat_lower in rel.lower():
                        matches.append((rel, 1, p.stem))
                        if len(matches) >= top_k:
                            break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Then: match file contents.
        cmd = [rg, "--no-heading", "-n", "--json"]
        cmd += ["-e", pattern] if is_regex else ["-F", pattern]
        cmd += [str(vault)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            for line in proc.stdout.splitlines():
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "match":
                    continue
                data = obj["data"]
                p = Path(data["path"]["text"])
                try:
                    rel = str(p.relative_to(vault))
                except ValueError:
                    rel = str(p)
                if not rel.endswith(".md"):
                    continue
                line_no = data["line_number"]
                text = data["lines"]["text"].rstrip("\n")
                matches.append((rel, line_no, text))
                if len(matches) >= top_k * 4:
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not matches:
        # SQL LIKE fallback over documents.content + title + path.
        like = f"%{pattern}%"
        rows = conn.execute(
            "SELECT id, path, content, mtime, title FROM documents "
            "WHERE content LIKE ? OR title LIKE ? OR path LIKE ?",
            (like, like, like),
        ).fetchall()
        for r in rows:
            pat_lower = pattern.lower()
            title_match = r["title"] and pat_lower in r["title"].lower()
            path_match = pat_lower in r["path"].lower()
            if title_match or path_match:
                matches.append((r["path"], 1, r["title"] or r["path"]))
            for i, line in enumerate(r["content"].splitlines(), start=1):
                if pat_lower in line.lower():
                    matches.append((r["path"], i, line))
                    if len(matches) >= top_k * 4:
                        break
            if len(matches) >= top_k * 4:
                break

    out: List[Candidate] = []
    for rel, line_no, line_text in matches[: top_k * 4]:
        doc = conn.execute(
            "SELECT id, title, mtime FROM documents WHERE path = ?", (rel,)
        ).fetchone()
        if doc is None:
            continue
        ch = _resolve_chunk(conn, doc["id"], line_no)
        if ch is None:
            chunk_id = -1
            heading_path = ""
            text = line_text
            start_line = line_no
        else:
            chunk_id = ch["id"]
            heading_path = ch["heading_path"] or ""
            text = ch["text"]
            start_line = ch["start_line"]
        out.append(
            Candidate(
                chunk_id=chunk_id,
                doc_id=doc["id"],
                doc_path=rel,
                doc_title=doc["title"] or rel,
                heading_path=heading_path,
                text=text,
                snippet=_make_snippet(line_text, pattern),
                score=1.0,
                doc_mtime=doc["mtime"] or 0,
                start_line=start_line,
            )
        )
        if len(out) >= top_k:
            break
    return out
