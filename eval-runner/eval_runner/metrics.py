"""Composite + aggregation metrics on top of rag.metrics primitives.

The per-row lexical/semantic metrics live in rag/rag/metrics.py and are
called directly inside the inference loop. Here we only define:

- composite RAG score (configurable weighted mix of RAGAS metrics)
- aggregate summary (means, win/lose counts, latency stats)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from eval_runner.config import RunConfig


def composite_rag_score(
    faithfulness: float,
    answer_relevancy: float,
    context_recall: float,
    cfg: RunConfig,
) -> float:
    """Weighted RAGAS-score with configurable weights.

    Current defaults are faithfulness-priority (0.6 / 0.2 / 0.2).
    BaseLine.ipynb and the historical defense deck used 0.4 / 0.4 / 0.2.
    NaN components are treated as zero so a failed component remains
    visible in the reduced score.
    """
    def _safe(x: float) -> float:
        return 0.0 if pd.isna(x) else float(x)

    return (
        cfg.rag_score_w_faithfulness * _safe(faithfulness)
        + cfg.rag_score_w_answer_relevancy * _safe(answer_relevancy)
        + cfg.rag_score_w_context_recall * _safe(context_recall)
    )


def aggregate_summary(per_row: pd.DataFrame, cfg: RunConfig) -> dict[str, Any]:
    """Reduce per-row DataFrame to a flat dict of scalars for wandb logging.

    Skips metrics whose source column is absent (e.g. RAGAS disabled).
    Keys are namespaced (lexical/, semantic/, ragas/, latency/, score/)
    so they group nicely in the wandb UI.
    """
    summary: dict[str, Any] = {"n_samples": len(per_row)}

    def _mean_if(col: str) -> float | None:
        if col not in per_row.columns:
            return None
        s = pd.to_numeric(per_row[col], errors="coerce")
        return float(s.mean()) if not s.dropna().empty else None

    # lexical (rag vs pure baseline)
    for side in ("rag", "pure"):
        for metric in ("f1", "precision", "recall"):
            v = _mean_if(f"{side}_{metric}")
            if v is not None:
                summary[f"lexical/{side}_{metric}"] = v
    if "rag_f1" in per_row.columns and "pure_f1" in per_row.columns:
        summary["lexical/rag_better_count"] = int(
            (per_row["rag_f1"] > per_row["pure_f1"]).sum()
        )
        summary["lexical/pure_better_count"] = int(
            (per_row["pure_f1"] > per_row["rag_f1"]).sum()
        )

    # semantic
    sim = _mean_if("answer_similarity")
    if sim is not None:
        summary["semantic/answer_similarity"] = sim

    # ragas
    for metric in ("faithfulness", "answer_relevancy", "context_recall"):
        v = _mean_if(metric)
        if v is not None:
            summary[f"ragas/{metric}"] = v

    # composite (only if all three ragas metrics ran)
    needed = {"ragas/faithfulness", "ragas/answer_relevancy", "ragas/context_recall"}
    if needed.issubset(summary.keys()):
        summary["score/rag_score"] = composite_rag_score(
            summary["ragas/faithfulness"],
            summary["ragas/answer_relevancy"],
            summary["ragas/context_recall"],
            cfg,
        )

    # latencies
    for side in ("rag", "pure"):
        col = f"latency_{side}"
        if col in per_row.columns:
            s = pd.to_numeric(per_row[col], errors="coerce")
            summary[f"latency/{side}_mean_s"] = float(s.mean())
            summary[f"latency/{side}_p50_s"] = float(s.quantile(0.5))
            summary[f"latency/{side}_p95_s"] = float(s.quantile(0.95))

    return summary
