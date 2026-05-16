"""Quality filters for the StackOverflow Q&A dataset."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from loguru import logger

JUNK_ANSWER_PATTERNS = [
    re.compile(r"^\s*(thanks?|thank you|thx)[\s.!]*$", re.IGNORECASE),
    re.compile(r"^\s*\+1[\s.!]*$"),
    re.compile(r"^\s*(me too|same here|bump|same problem)[\s.!]*$", re.IGNORECASE),
    re.compile(r"^\s*(see (the )?(edit|update|comment))[\s.!:]*$", re.IGNORECASE),
    re.compile(r"^\s*(edit|update)\s*[:.]?\s*$", re.IGNORECASE),
]


@dataclass(frozen=True)
class FilterConfig:
    min_score: int = 5
    min_question_chars: int = 50
    max_question_chars: int = 4000
    min_answer_chars: int = 100
    max_answer_chars: int = 6000


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    dropped_score: int = 0
    dropped_q_too_short: int = 0
    dropped_q_too_long: int = 0
    dropped_a_too_short: int = 0
    dropped_a_too_long: int = 0
    dropped_junk: int = 0
    dropped_empty: int = 0


def _is_junk_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    for pattern in JUNK_ANSWER_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


@dataclass
class Pair:
    question: str
    answer: str
    score: int
    context: str = ""
    is_adversarial: bool = False


def filter_pairs(
    pairs: Iterable[tuple[str, str, int]],
    config: FilterConfig | None = None,
) -> tuple[list[Pair], FilterStats]:
    """Apply quality filters to (question, answer, score) triples.

    Returns the kept pairs plus a stats record describing what was dropped.
    Inputs are assumed to be already-cleaned markdown (call `html_to_markdown`
    first).
    """
    cfg = config or FilterConfig()
    stats = FilterStats()
    kept: list[Pair] = []

    for q, a, score in pairs:
        stats.total += 1

        if not q or not a:
            stats.dropped_empty += 1
            continue

        if score < cfg.min_score:
            stats.dropped_score += 1
            continue

        q_len = len(q)
        if q_len < cfg.min_question_chars:
            stats.dropped_q_too_short += 1
            continue
        if q_len > cfg.max_question_chars:
            stats.dropped_q_too_long += 1
            continue

        a_len = len(a)
        if a_len < cfg.min_answer_chars:
            stats.dropped_a_too_short += 1
            continue
        if a_len > cfg.max_answer_chars:
            stats.dropped_a_too_long += 1
            continue

        if _is_junk_answer(a):
            stats.dropped_junk += 1
            continue

        kept.append(Pair(question=q, answer=a, score=score))
        stats.kept += 1

    logger.info(
        "filter stats: total={total} kept={kept} "
        "dropped(score={s} q<{qmin}={qs} q>{qmax}={ql} "
        "a<{amin}={as_} a>{amax}={al} junk={j} empty={e})",
        total=stats.total,
        kept=stats.kept,
        s=stats.dropped_score,
        qmin=cfg.min_question_chars,
        qs=stats.dropped_q_too_short,
        qmax=cfg.max_question_chars,
        ql=stats.dropped_q_too_long,
        amin=cfg.min_answer_chars,
        as_=stats.dropped_a_too_short,
        amax=cfg.max_answer_chars,
        al=stats.dropped_a_too_long,
        j=stats.dropped_junk,
        e=stats.dropped_empty,
    )
    return kept, stats
