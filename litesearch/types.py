from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Candidate:
    chunk_id: int
    doc_id: int
    doc_path: str
    doc_title: str
    heading_path: str
    text: str
    snippet: str
    score: float
    doc_mtime: int
    start_line: int
    vec: Optional[List[float]] = field(default=None, repr=False)
