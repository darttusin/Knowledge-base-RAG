"""Load the evaluation split.

Lifted from notebooks/BaseLine.ipynb (cells 42-43) so the eval split
is reproducible: same source CSV, same filter, same random seed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from eval_runner.config import RunConfig

RAW_COLUMNS = ("question_body", "answer_body", "answer_score")


def load_eval_dataset(cfg: RunConfig) -> pd.DataFrame:
    """Return a `pd.DataFrame` with columns `question`, `answer`.

    Steps mirror the production notebook:
      1. read stackoverflow CSV
      2. rename to canonical `question` / `answer`
      3. drop rows with missing fields
      4. filter by `answer_score >= eval_min_answer_score`
      5. sample `eval_sample_size` rows with `eval_seed`
    """
    path = Path(cfg.eval_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"eval CSV not found: {path}")

    logger.info("loading eval CSV from {p}", p=path)
    df = pd.read_csv(path)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. found: {list(df.columns)}"
        )

    df = df[list(RAW_COLUMNS)].dropna(subset=["question_body", "answer_body"])
    df["answer_score"] = (
        pd.to_numeric(df["answer_score"], errors="coerce").fillna(0).astype(int)
    )
    df = df.rename(columns={"question_body": "question", "answer_body": "answer"})

    before = len(df)
    df = df[df["answer_score"] >= cfg.eval_min_answer_score]
    logger.info(
        "filtered by min_answer_score={s}: {after}/{before} rows",
        s=cfg.eval_min_answer_score,
        after=len(df),
        before=before,
    )

    if cfg.eval_sample_size < len(df):
        df = df.sample(cfg.eval_sample_size, random_state=cfg.eval_seed)
        logger.info(
            "sampled {n} rows (seed={seed})",
            n=cfg.eval_sample_size,
            seed=cfg.eval_seed,
        )
    else:
        logger.info(
            "eval_sample_size={n} >= available={a}, using all rows",
            n=cfg.eval_sample_size,
            a=len(df),
        )

    return df.reset_index(drop=True)[["question", "answer"]]
