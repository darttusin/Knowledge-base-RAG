"""Question-level deduplication.

StackOverflow has many near-duplicate questions ("How do I X?", "How to X
in PyTorch?"). We normalize the question to a canonical form and keep
only the highest-scoring answer per group.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from loguru import logger

from dataset_prep.filtering import Pair

CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
URL_RE = re.compile(r"https?://\S+")
NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Reduce a question to a canonical form for duplicate detection.

    Strips code blocks, URLs, punctuation, casing, and whitespace.
    """
    text = unicodedata.normalize("NFKD", text)
    text = CODE_BLOCK_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def deduplicate(pairs: Iterable[Pair]) -> list[Pair]:
    """Keep one pair per normalized question — the one with the highest score."""
    best: dict[str, Pair] = {}
    total = 0
    for pair in pairs:
        total += 1
        key = normalize_question(pair.question)
        if not key:
            continue
        existing = best.get(key)
        if existing is None or pair.score > existing.score:
            best[key] = pair

    kept = list(best.values())
    logger.info(
        "dedup: total={total} unique={unique} dropped={dropped}",
        total=total,
        unique=len(kept),
        dropped=total - len(kept),
    )
    return kept
