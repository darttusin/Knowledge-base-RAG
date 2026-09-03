"""Extended W&B logging.

Every field of RunConfig becomes part of `wandb.config`. This records the
runner's configurable subset, not complete model/corpus/prompt provenance.
The summary dict goes in as flat scalars; the per-row dataframe goes in as a
sortable Table; low-faithfulness rows go in a separate "hallucinations" table.

RunConfig currently includes generator/judge key fields and per-row data
can include private context. Do not use real secrets or sensitive corpora
until this module implements redaction and an explicit no-tracking mode.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import wandb
from loguru import logger

from eval_runner.config import auto_run_name
from eval_runner.runner import EvalResult

# wandb Tables don't handle list-typed columns gracefully — serialize
# `contexts` to a readable string before logging.
CONTEXTS_PREVIEW_CHARS = 600


def _serialize_for_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "contexts" in out.columns:
        out["contexts"] = out["contexts"].apply(
            lambda chunks: "\n\n---\n\n".join(
                (c if isinstance(c, str) else str(c)) for c in (chunks or [])
            )[:CONTEXTS_PREVIEW_CHARS * 5]
        )
    return out


def log_to_wandb(result: EvalResult, *, api_key: str | None = None) -> str:
    """Push the eval result to wandb. Returns the run URL."""
    cfg = result.config
    if api_key:
        wandb.login(key=api_key)

    run_name = cfg.wandb_run_name or auto_run_name(cfg)
    config_dict = asdict(cfg)

    run = wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config=config_dict,
        tags=cfg.wandb_tags or None,
        notes=cfg.wandb_notes or None,
    )
    logger.info("wandb run started: {url}", url=run.url)

    # scalar metrics — already namespaced (lexical/, semantic/, ragas/, …)
    wandb.log(result.summary)

    # full per-row table
    table_df = _serialize_for_table(result.per_row)
    wandb.log({"per_row_table": wandb.Table(dataframe=table_df)})

    # bad cases — low faithfulness rows for manual inspection
    if "faithfulness" in result.per_row.columns:
        bad = result.per_row[result.per_row["faithfulness"] < 0.5]
        if not bad.empty:
            wandb.log(
                {"hallucinations": wandb.Table(dataframe=_serialize_for_table(bad))}
            )
            logger.info("logged {n} low-faithfulness rows", n=len(bad))

    # cases where rag clearly LOST to baseline — useful for failure analysis
    if {"rag_f1", "pure_f1"}.issubset(result.per_row.columns):
        losses = result.per_row[result.per_row["pure_f1"] > result.per_row["rag_f1"] + 0.1]
        if not losses.empty:
            wandb.log(
                {"rag_lost_to_baseline": wandb.Table(dataframe=_serialize_for_table(losses))}
            )

    url = run.url
    run.finish()
    return url
