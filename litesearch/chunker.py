from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    heading_path: str
    start_line: int
    end_line: int

    def with_offset(self, offset: int) -> Chunk:
        return Chunk(
            text=self.text,
            heading_path=self.heading_path,
            start_line=self.start_line + offset,
            end_line=self.end_line + offset,
        )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_headings(body: str) -> List[tuple[int, int, int, str, str]]:
    lines = body.splitlines()
    sections: List[tuple[int, int, int, str, str]] = []

    headings: List[tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings or headings[0][0] > 0:
        end_idx = headings[0][0] if headings else len(lines)
        text = "\n".join(lines[:end_idx]).strip()
        if text:
            sections.append((1, end_idx if end_idx else 1, 0, "", text))

    for idx, (line_idx, depth, htext) in enumerate(headings):
        if idx + 1 < len(headings):
            next_idx = headings[idx + 1][0]
        else:
            next_idx = len(lines)
        section_lines = lines[line_idx:next_idx]
        text = "\n".join(section_lines).strip()
        if text:
            sections.append((line_idx + 1, next_idx, depth, htext, text))

    return sections


def _build_heading_path(stack: List[tuple[int, str]], depth: int, htext: str) -> str:
    while stack and stack[-1][0] >= depth:
        stack.pop()
    if depth > 0:
        stack.append((depth, htext))
    return " > ".join(h for _, h in stack)


def _split_long(text: str, soft_max: int) -> List[str]:
    if len(text) <= soft_max:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    out: List[str] = []
    buf = ""
    for p in paragraphs:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) > soft_max and buf:
            out.append(buf)
            buf = p
        else:
            buf = candidate
    if buf:
        out.append(buf)
    final: List[str] = []
    for piece in out:
        while len(piece) > soft_max * 2:
            final.append(piece[:soft_max])
            piece = piece[soft_max:]
        final.append(piece)
    return final


def chunk_body(body: str, soft_max_chars: int = 1500) -> List[Chunk]:
    sections = _split_headings(body)
    stack: List[tuple[int, str]] = []
    chunks: List[Chunk] = []

    for start_line, end_line, depth, htext, section_text in sections:
        path = _build_heading_path(stack, depth, htext) if depth > 0 else " > ".join(h for _, h in stack)
        pieces = _split_long(section_text, soft_max_chars)
        if len(pieces) == 1:
            chunks.append(Chunk(pieces[0], path, start_line, end_line))
        else:
            total = sum(len(p) for p in pieces) or 1
            cursor = start_line
            for p in pieces:
                span = max(1, round((end_line - start_line + 1) * len(p) / total))
                ce = min(end_line, cursor + span - 1)
                chunks.append(Chunk(p, path, cursor, ce))
                cursor = ce + 1
    return chunks
