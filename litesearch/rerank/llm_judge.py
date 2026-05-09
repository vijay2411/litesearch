from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel

from ..types import Candidate

log = logging.getLogger("litesearch.rerank.llm")


class _Ranking(BaseModel):
    order: List[int]


_SYSTEM_PROMPT = """You are a search-result reranker.

You receive a query and a numbered list of candidate passages. Your job is
to ORDER the candidates from most relevant to least relevant for the query.

Rules:
- Read every candidate before deciding the order.
- Tied or near-tied candidates: pick one — the order must be a strict
  permutation, no ties.
- Reward candidates that directly answer the query over candidates that
  just mention the same keywords.
- A candidate that defines or directly addresses the query topic ranks
  ABOVE one that only references it in passing.

Respond with a single JSON object: {"order": [<id>, ...]}. No prose, no
markdown fences. The list contains every candidate id you were given,
exactly once, best first."""

_USER_PROMPT = """Query: {query}

You will rank exactly {n} candidates below. Your "order" array MUST contain
exactly {n} integer ids — every candidate id from 0 to {last} included once,
no duplicates, no extras.

Candidates:
{candidates}"""


def _rank_via_gemini(
    system: str,
    user: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> _Ranking | None:
    from google import genai

    if not api_key:
        log.warning("Gemini API key not configured")
        return None

    client = genai.Client(api_key=api_key)
    cfg: dict = {
        "system_instruction": system,
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": _Ranking,
    }
    try:
        from google.genai import types as genai_types
        cfg["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    try:
        resp = client.models.generate_content(
            model=model, contents=user, config=cfg,
        )
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return None

    raw = (resp.text or "").strip()
    if not raw:
        log.warning("Gemini returned empty content")
        return None
    try:
        return _Ranking.model_validate_json(raw)
    except Exception as e:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m is not None:
            try:
                return _Ranking.model_validate_json(m.group(0))
            except Exception:
                pass
        log.warning("Gemini response parse failed: %s\nraw=%s", e, raw[:300])
        return None


def _rank_via_ollama(
    system: str,
    user: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma4",
) -> _Ranking | None:
    import ollama

    client = ollama.Client(host=ollama_url)
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": 0, "num_predict": 1200},
            keep_alive="30m",
        )
    except Exception as e:
        log.warning("Ollama LLM judge failed: %s", e)
        return None

    raw = resp["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m is None:
        if not raw.strip():
            log.warning("Ollama LLM judge: empty response")
        return None
    try:
        return _Ranking.model_validate_json(m.group(0))
    except Exception as e:
        log.warning("Ollama LLM judge parse failed: %s\nraw=%s", e, raw[:300])
        return None


def rerank_llm(
    query: str,
    candidates: List[Candidate],
    top_k: int,
    *,
    backend: str = "gemini",
    gemini_api_key: str = "",
    gemini_model: str = "gemini-2.5-flash",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "gemma4",
) -> List[Candidate]:
    if not candidates:
        return []

    if backend == "gemini":
        pool_size = 15
        truncate = 800
    else:
        pool_size = 8
        truncate = 400

    pool = candidates[:pool_size]
    blocks = "\n\n".join(
        f"id={i}: {c.text[:truncate].strip()}" for i, c in enumerate(pool)
    )
    user_msg = _USER_PROMPT.format(
        query=query, candidates=blocks, n=len(pool), last=len(pool) - 1,
    )

    if backend == "gemini":
        parsed = _rank_via_gemini(_SYSTEM_PROMPT, user_msg, gemini_api_key, gemini_model)
    elif backend == "ollama":
        parsed = _rank_via_ollama(_SYSTEM_PROMPT, user_msg, ollama_url, ollama_model)
    else:
        log.warning("Unknown llm_judge_backend %r — passing through", backend)
        parsed = None

    if parsed is None:
        return candidates[:top_k]

    seen: set[int] = set()
    ordered_ids: List[int] = []
    for i in parsed.order:
        if 0 <= i < len(pool) and i not in seen:
            seen.add(i)
            ordered_ids.append(i)
    for i in range(len(pool)):
        if i not in seen:
            ordered_ids.append(i)

    n = len(ordered_ids)
    out: List[Candidate] = []
    for rank, i in enumerate(ordered_ids[:top_k]):
        c = pool[i]
        c.score = 1.0 if n == 1 else 1.0 - rank / (n - 1)
        out.append(c)
    return out
