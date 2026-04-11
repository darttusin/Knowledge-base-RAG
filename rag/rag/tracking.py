from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import wandb


def generate_run_name(config: dict[str, Any]) -> str:
    gen_model = config.get("gen_model", "").split("/")[-1]
    gen_model = (
        gen_model.replace("gemini-", "Gem")
        .replace("-flash", "Fl")
        .replace("-pro", "Pro")
    )

    embed_model = config.get("embed_model", "").split("/")[-1]
    embed_model = embed_model.replace("-en-v1.5", "").replace("snowflake-", "Snow")

    sz = config.get("chunk_size", 0)
    ov = config.get("chunk_overlap", 0)
    k = config.get("top_k", 0)

    return f"{gen_model}_{embed_model}_sz{sz}_ov{ov}_k{k}"


@dataclass
class ExperimentResult:
    rag_score: float
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    ground_summary: dict[str, Any] = field(default_factory=dict)
    rag_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    reranking_results: pd.DataFrame | None = None


def log_experiment(
    *,
    project: str,
    config: dict[str, Any],
    result: ExperimentResult,
    wandb_api_key: str | None = None,
) -> None:
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run_name = generate_run_name(config)

    run = wandb.init(project=project, name=run_name, config=config)

    wandb.log(
        {
            "global/rag_score": result.rag_score,
            "global/faithfulness": result.faithfulness,
            "global/answer_relevancy": result.answer_relevancy,
            "global/context_recall": result.context_recall,
            "ground/rag_mean_f1": result.ground_summary["rag_mean_f1"],
            "ground/pure_mean_f1": result.ground_summary["pure_mean_f1"],
            "ground/answer_similarity": result.ground_summary["answer_similarity"],
            "ground/rag_better_count": result.ground_summary["rag_better_count"],
        }
    )

    wandb.log({"rag_results_table": wandb.Table(dataframe=result.rag_results)})

    if result.reranking_results is not None:
        wandb.log(
            {
                "reranking/analysis_table": wandb.Table(
                    dataframe=result.reranking_results
                ),
            }
        )

    bad_cases = result.rag_results[result.rag_results["faithfulness"] < 0.5]
    if not bad_cases.empty:
        wandb.log({"hallucinations": wandb.Table(dataframe=bad_cases)})

    run.finish()
