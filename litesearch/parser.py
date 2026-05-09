from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter


@dataclass
class ParsedDoc:
    title: str
    frontmatter_json: str
    body: str
    body_offset_lines: int


_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def parse_markdown(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8", errors="replace")
    post = frontmatter.loads(raw)
    body = post.content
    fm = post.metadata or {}
    fm_json = json.dumps(fm, default=str, ensure_ascii=False)

    body_offset_lines = 0
    if raw.startswith("---"):
        m = re.search(r"\n---\s*\n", raw)
        if m:
            body_offset_lines = raw[: m.end()].count("\n")

    m = _H1_RE.search(body)
    title = m.group(1).strip() if m else path.stem
    return ParsedDoc(
        title=title,
        frontmatter_json=fm_json,
        body=body,
        body_offset_lines=body_offset_lines,
    )
