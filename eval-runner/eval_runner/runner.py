"""End-to-end evaluation runner.

Orchestrates: load eval split → build pipeline → loop over questions
(measuring latency) → per-row lexical metrics → semantic similarity →
RAGAS (subprocess venv) → aggregate → return EvalResult.

The result object holds everything needed for downstream logging or
local inspection. Tracking to wandb is a separate step (see
eval_runner.tracking) so unit tests / dry runs can skip it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.evaluation import run_ragas_evaluation
from rag.metrics import (
    embedding_similarity,
    squad_f1,
    squad_precision,
    squad_recall,
)

from eval_runner.config import RunConfig
from eval_runner.data import load_eval_dataset
from eval_runner.metrics import aggregate_summary
from eval_runner.pipeline import Pipeline, build_pipeline


@dataclass
class EvalResult:
    config: RunConfig
    per_row: pd.DataFrame
    summary: dict[str, Any]


def _run_inference_loop(
    eval_df: pd.DataFrame,
    pipeline: Pipeline,
    cfg: RunConfig,
) -> pd.DataFrame:
    """Run rag_fn + pure_fn on each row, capturing answers, contexts, latencies."""
    records: list[dict[str, Any]] = []

    for row in tqdm(eval_df.itertuples(index=False), total=len(eval_df), desc="inference"):
        q = row.question
        gold = row.answer

        t0 = time.perf_counter()
        try:
            rag_answer, contexts = pipeline.rag_fn(q)
        except Exception as exc:
            logger.warning("rag_fn failed on q={q!r}: {e}", q=q[:80], e=exc)
            rag_answer, contexts = "", []
        t_rag = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            pure_answer = pipeline.pure_fn(q)
        except Exception as exc:
            logger.warning("pure_fn failed on q={q!r}: {e}", q=q[:80], e=exc)
            pure_answer = ""
        t_pure = time.perf_counter() - t0

        rec: dict[str, Any] = {
            "question": q,
            "gold": gold,
            "rag_answer": rag_answer,
            "pure_answer": pure_answer,
            "contexts": contexts,
            "latency_rag": t_rag,
            "latency_pure": t_pure,
        }

        if cfg.compute_lexical:
            rec["rag_precision"] = squad_precision(rag_answer, gold)
            rec["rag_recall"] = squad_recall(rag_answer, gold)
            rec["rag_f1"] = squad_f1(rag_answer, gold)
            rec["pure_precision"] = squad_precision(pure_answer, gold)
            rec["pure_recall"] = squad_recall(pure_answer, gold)
            rec["pure_f1"] = squad_f1(pure_answer, gold)

        records.append(rec)

    return pd.DataFrame(records)


def _add_semantic_similarity(per_row: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """rag_answer ↔ pure_answer cosine similarity via a SEPARATE embed model.

    Using a different model than the retriever's embedding is intentional:
    avoids tautology where the metric model "agrees with itself".
    """
    logger.info("loading semantic eval embed model: {m}", m=cfg.eval_embedding_model)
    eval_embed = SentenceTransformer(cfg.eval_embedding_model)
    sims = [
        embedding_similarity(r, p, eval_embed)
        for r, p in zip(per_row["rag_answer"], per_row["pure_answer"], strict=True)
    ]
    per_row = per_row.copy()
    per_row["answer_similarity"] = sims
    return per_row


def _maybe_add_ragas(per_row: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Append faithfulness / answer_relevancy / context_recall to per_row."""
    if not cfg.judge_api_url:
        logger.error("RAGAS enabled but judge_api_url empty — skipping")
        return per_row

    ragas_input = pd.DataFrame(
        {
            "question": per_row["question"],
            "answer": per_row["rag_answer"],
            "contexts": per_row["contexts"],
            "ground_truth": per_row["gold"],
        }
    )
    logger.info("invoking RAGAS subprocess in an isolated, reusable venv")
    ragas_df = run_ragas_evaluation(
        df_prepared=ragas_input,
        api_url=cfg.judge_api_url,
        api_key=cfg.judge_api_key,
        judge_model=cfg.judge_model,
        embed_model=cfg.embedding_model,
    )
    if ragas_df.empty:
        logger.warning("RAGAS returned empty dataframe — skipping metrics")
        return per_row

    per_row = per_row.copy()
    for col in ("faithfulness", "answer_relevancy", "context_recall"):
        if col in ragas_df.columns:
            per_row[col] = ragas_df[col].values
        else:
            logger.warning("ragas dataframe missing column {c}", c=col)
    return per_row


def run_evaluation(cfg: RunConfig) -> EvalResult:
    """Run the full evaluation pipeline and return its in-memory results.

    This can download models, open the configured Chroma index, call generator
    and judge endpoints, and create or refresh the local RAGAS environment.
    W&B logging is performed separately by the CLI.
    """
    logger.info("starting eval run: {n} samples", n=cfg.eval_sample_size)
    eval_df = load_eval_dataset(cfg)
    pipeline = build_pipeline(cfg)

    per_row = _run_inference_loop(eval_df, pipeline, cfg)

    if cfg.compute_semantic:
        per_row = _add_semantic_similarity(per_row, cfg)

    if cfg.compute_ragas:
        per_row = _maybe_add_ragas(per_row, cfg)

    summary = aggregate_summary(per_row, cfg)
    logger.info("eval done. summary: {s}", s=summary)

    return EvalResult(config=cfg, per_row=per_row, summary=summary)
