"""End-to-end dataset preparation pipeline.

Reads the raw StackOverflow CSV, cleans HTML, filters, deduplicates,
splits, retrieves PyTorch documentation context for each question, and
adds adversarial refusal-target examples. Writes the legacy context-enriched
train/val JSONL format consumed by the current training reader; serving prompt
compatibility still needs separate contract handling.

Output schema (one JSON object per line):
    {
        "question": "...",
        "answer": "...",
        "score": 39,
        "context": "...",         // retrieved top-k chunks joined
        "is_adversarial": false   // true for synthetic refusal examples
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from dataset_prep.cleaning import html_to_markdown
from dataset_prep.dedup import deduplicate
from dataset_prep.filtering import FilterConfig, Pair, filter_pairs
from dataset_prep.retrieval import (
    RetrievalConfig,
    enrich_and_augment,
    load_retrieval_context,
)
from dataset_prep.splitting import stratified_split

REQUIRED_COLUMNS = ("question_body", "answer_body", "answer_score")


@dataclass
class PipelineConfig:
    csv_path: Path
    output_dir: Path
    val_fraction: float = 0.05
    seed: int = 42
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)


def _load_raw(csv_path: Path) -> pd.DataFrame:
    logger.info("reading {path}", path=csv_path)
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. found: {list(df.columns)}"
        )
    df = df[list(REQUIRED_COLUMNS)].dropna(subset=["question_body", "answer_body"])
    df["answer_score"] = pd.to_numeric(df["answer_score"], errors="coerce").fillna(0).astype(int)
    logger.info("loaded {n} non-empty rows", n=len(df))
    return df


def _clean_rows(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    cleaned: list[tuple[str, str, int]] = []
    for q_html, a_html, score in tqdm(
        zip(df["question_body"], df["answer_body"], df["answer_score"], strict=True),
        total=len(df),
        desc="html→markdown",
    ):
        q_md = html_to_markdown(q_html)
        a_md = html_to_markdown(a_html)
        cleaned.append((q_md, a_md, int(score)))
    return cleaned


def _write_jsonl(path: Path, pairs: list[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            json.dump(
                {
                    "question": p.question,
                    "answer": p.answer,
                    "score": p.score,
                    "context": p.context,
                    "is_adversarial": p.is_adversarial,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    logger.info("wrote {n} rows → {path}", n=len(pairs), path=path)


def run_pipeline(config: PipelineConfig) -> dict[str, int]:
    """Run the full RAG-aware prep pipeline and return summary counts."""
    df = _load_raw(config.csv_path)
    cleaned = _clean_rows(df)
    filtered, _ = filter_pairs(cleaned, config.filter_config)
    deduped = deduplicate(filtered)
    train_raw, val_raw = stratified_split(deduped, config.val_fraction, config.seed)

    logger.info("loading retrieval context (chromadb + embedding model)")
    retrieval_ctx = load_retrieval_context(config.retrieval_config)

    logger.info("enriching train split with context + adversarial")
    train = enrich_and_augment(train_raw, retrieval_ctx)
    logger.info("enriching val split with context + adversarial")
    val = enrich_and_augment(val_raw, retrieval_ctx)

    _write_jsonl(config.output_dir / "train.jsonl", train)
    _write_jsonl(config.output_dir / "val.jsonl", val)

    summary = {
        "raw": len(df),
        "after_filter": len(filtered),
        "after_dedup": len(deduped),
        "train": len(train),
        "val": len(val),
        "train_adversarial": sum(1 for p in train if p.is_adversarial),
        "val_adversarial": sum(1 for p in val if p.is_adversarial),
    }
    logger.info("pipeline done: {summary}", summary=summary)
    return summary
