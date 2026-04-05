from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Lexical (SQuAD-style token overlap)
# ---------------------------------------------------------------------------


def _normalize(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return [t for t in text.split() if t]


def _token_overlap(pred: str, gold: str) -> tuple[int, int, int]:
    p_tokens = _normalize(pred)
    g_tokens = _normalize(gold)
    if not p_tokens and not g_tokens:
        return 0, 0, 0
    common: dict[str, int] = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    return overlap, len(p_tokens), len(g_tokens)


def squad_precision(pred: str, gold: str) -> float:
    overlap, p_len, g_len = _token_overlap(pred, gold)
    if p_len == 0 and g_len == 0:
        return 1.0
    if p_len == 0:
        return 0.0
    return overlap / p_len


def squad_recall(pred: str, gold: str) -> float:
    overlap, p_len, g_len = _token_overlap(pred, gold)
    if p_len == 0 and g_len == 0:
        return 1.0
    if g_len == 0:
        return 0.0
    return overlap / g_len


def squad_f1(pred: str, gold: str) -> float:
    p = squad_precision(pred, gold)
    r = squad_recall(pred, gold)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Semantic (embedding cosine similarity)
# ---------------------------------------------------------------------------


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


def embedding_similarity(
    pred: str,
    gold: str,
    model: SentenceTransformer,
) -> float:
    vecs = model.encode([pred, gold])
    return cosine(vecs[0], vecs[1])
