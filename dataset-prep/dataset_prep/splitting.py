"""Train/validation split stratified by answer score."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from sklearn.model_selection import train_test_split

from dataset_prep.filtering import Pair

SCORE_BINS: tuple[tuple[int, int, str], ...] = (
    (0, 10, "low"),
    (10, 50, "mid"),
    (50, 200, "high"),
    (200, 10**9, "top"),
)


def _bin_score(score: int) -> str:
    for lo, hi, label in SCORE_BINS:
        if lo <= score < hi:
            return label
    return SCORE_BINS[-1][2]


def stratified_split(
    pairs: Sequence[Pair],
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[Pair], list[Pair]]:
    """Split pairs into train/val, stratified by binned `answer_score`.

    A small bin (fewer than 2 samples) is folded into the nearest non-empty
    bin so `train_test_split` doesn't fail.
    """
    if not pairs:
        return [], []

    bins = [_bin_score(p.score) for p in pairs]

    counts: dict[str, int] = {}
    for b in bins:
        counts[b] = counts.get(b, 0) + 1
    rare = {b for b, c in counts.items() if c < 2}
    if rare:
        logger.warning("collapsing rare score bins into 'low': {rare}", rare=rare)
        bins = ["low" if b in rare else b for b in bins]

    train_pairs, val_pairs = train_test_split(
        list(pairs),
        test_size=val_fraction,
        random_state=seed,
        stratify=bins,
    )
    logger.info(
        "split: train={train} val={val} (val_fraction={frac})",
        train=len(train_pairs),
        val=len(val_pairs),
        frac=val_fraction,
    )
    return train_pairs, val_pairs
