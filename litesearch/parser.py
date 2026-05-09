from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import frontmatter


@dataclass
class ParsedDoc:
    title: str
    frontmatter_json: str
    body: str
    body_offset_lines: int


_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)

_MARKDOWN_EXTS = {".md", ".markdown", ".mdx", ".mdown"}


def parse_file(path: Path) -> ParsedDoc:
    """Parse any text file. Markdown gets frontmatter extraction; everything
    else is treated as plain text with title derived from filename."""
    if path.suffix.lower() in _MARKDOWN_EXTS:
        return _parse_markdown(path)
    return _parse_plaintext(path)


def _parse_markdown(path: Path) -> ParsedDoc:
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


def _parse_plaintext(path: Path) -> ParsedDoc:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return ParsedDoc(
        title=path.stem,
        frontmatter_json="{}",
        body=raw,
        body_offset_lines=0,
    )


def parse_jsonl(path: Path, text_field: str = "text", title_field: str = "title") -> Iterator[ParsedDoc]:
    """Yield one ParsedDoc per line in a JSONL file.

    Each JSON object must have a text field. Title is derived from
    title_field if present, otherwise from the line number.
    """
    for i, line in enumerate(path.open(encoding="utf-8", errors="replace"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get(text_field)
        if not text:
            continue
        title = obj.get(title_field) or f"{path.stem}:{i}"
        yield ParsedDoc(
            title=str(title),
            frontmatter_json=json.dumps(obj, default=str, ensure_ascii=False),
            body=str(text),
            body_offset_lines=0,
        )


# Keep the old name as an alias for backward compat
parse_markdown = _parse_markdown
