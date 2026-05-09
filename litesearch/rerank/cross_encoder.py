from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import List

from ..types import Candidate

log = logging.getLogger("litesearch.rerank.ce")

_SCORE_LINE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+(.*)$")


def _resolve_cli(cli: str) -> str | None:
    if "/" in cli:
        return cli
    return shutil.which(cli)


def rerank_cross_encoder(
    query: str,
    candidates: List[Candidate],
    top_k: int,
    cli: str = "rerank",
) -> List[Candidate]:
    if not candidates:
        return []
    resolved = _resolve_cli(cli)
    if not resolved:
        log.warning("rerank CLI not found at %s — passing through", cli)
        return candidates[:top_k]

    texts = [c.text.replace("\n", " ").strip()[:2000] for c in candidates]
    cmd = [resolved, query, *texts]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.warning("rerank CLI failed: %s — passing through", e)
        return candidates[:top_k]

    if proc.returncode != 0:
        log.warning("rerank CLI nonzero exit: %s", proc.stderr.strip()[:200])
        return candidates[:top_k]

    by_text = {t: i for i, t in enumerate(texts)}
    scored: list[tuple[float, int]] = []
    for line in proc.stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        score_str: str | None = None
        text_part: str | None = None
        if "\t" in line:
            score_str, text_part = line.split("\t", 1)
        else:
            m = _SCORE_LINE.match(line)
            if m:
                score_str, text_part = m.group(1), m.group(2)
        if score_str is None:
            continue
        try:
            score = float(score_str)
        except ValueError:
            continue
        if text_part is None:
            continue
        idx = by_text.get(text_part.strip())
        if idx is None:
            for t, i in by_text.items():
                if t.startswith(text_part.strip()[:80]):
                    idx = i
                    break
        if idx is not None:
            scored.append((score, idx))

    if not scored:
        log.warning("rerank CLI produced no parseable lines")
        return candidates[:top_k]

    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Candidate] = []
    seen: set[int] = set()
    for score, idx in scored:
        if idx in seen:
            continue
        seen.add(idx)
        c = candidates[idx]
        c.score = float(score)
        out.append(c)
        if len(out) >= top_k:
            break
    return out
