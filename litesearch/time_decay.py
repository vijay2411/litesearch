from __future__ import annotations

import math
import time
from typing import List

from .types import Candidate


def apply_time_decay(
    candidates: List[Candidate],
    half_life_days: int,
    weight: float,
    now: int | None = None,
) -> List[Candidate]:
    if weight <= 0 or not candidates:
        return candidates
    weight = max(0.0, min(1.0, weight))
    now = now or int(time.time())
    ln2 = math.log(2)
    hl = max(1, half_life_days)
    for c in candidates:
        age_days = max(0.0, (now - (c.doc_mtime or 0)) / 86400.0)
        decay = math.exp(-ln2 * age_days / hl)
        c.score = c.score * (1 - weight) + decay * weight
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates
