"""All knobs of a single evaluation run.

The whole point of this dataclass is reproducibility: every field that
can change between runs lives here. `asdict(RunConfig)` becomes
`wandb.config`, so any future comparison in wandb knows exactly what
produced each datapoint.

To compare "base model vs LoRA": run twice with different `llm_model` /
`llm_api_url`. The vLLM server should be started with `--enable-lora
--lora-modules <alias>=<adapter-path>`, then point `llm_model` at the
alias for the LoRA run, and at the base model name for the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RetrieverType = Literal["vanilla", "rerank", "query_transform"]


@dataclass
class RunConfig:
    # === Retriever ===
    retriever_type: RetrieverType = "vanilla"
    top_k: int = 5
    fetch_k: int = 20  # only consulted for rerank / query_transform
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    chroma_path: str = "data/chromadb"
    chroma_collection: str = "docs_fast"
    device: str = "auto"  # auto / cpu / cuda / mps

    # === Generator LLM ===
    # For LoRA on/off: change llm_model to the vLLM --lora-modules alias
    # for the LoRA run, and to the bare base model for the baseline.
    llm_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    llm_api_url: str = ""
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0

    # === Judge LLM (for RAGAS) ===
    # judge_api_key defaults to "EMPTY" because vLLM endpoints don't
    # validate the key but langchain_openai.ChatOpenAI requires a
    # non-empty string. Override via the JSON config file if you point
    # the judge at a cloud provider (OpenAI, OpenRouter, …).
    judge_model: str = "gpt-4o-mini"
    judge_api_url: str = ""
    judge_api_key: str = "EMPTY"

    # === Eval dataset ===
    eval_csv_path: str = "data/stackoverflow-pytorch.csv"
    eval_sample_size: int = 100
    eval_min_answer_score: int = 1
    eval_seed: int = 42

    # === Embedding model for semantic similarity ===
    eval_embedding_model: str = "Snowflake/snowflake-arctic-embed-m"

    # === What to compute ===
    compute_lexical: bool = True  # squad_f1 / precision / recall
    compute_semantic: bool = True  # answer↔answer cosine sim
    compute_ragas: bool = True  # faithfulness, answer_relevancy, context_recall

    # === Composite RAG score weights ===
    # Faithfulness-priority: for a RAG assistant over a private knowledge
    # base, a confidently-wrong (hallucinated) API answer is worse than a
    # slightly-less-relevant but grounded one — groundedness is the whole
    # point of RAG. Hence faithfulness gets the dominant weight. The
    # original BaseLine notebook used 0.4/0.4/0.2; we shifted to faith
    # priority after the v2 evaluation showed a faithfulness↔relevancy
    # trade-off. Always report a weight-sensitivity analysis alongside the
    # headline number (see scripts/recompute_score.py).
    rag_score_w_faithfulness: float = 0.6
    rag_score_w_answer_relevancy: float = 0.2
    rag_score_w_context_recall: float = 0.2

    # === Tracking ===
    wandb_project: str = "pytorch-rag-eval"
    wandb_run_name: str | None = None  # auto-generated if None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str = ""

    # === Free-form metadata for the experiment ===
    description: str = ""
    # e.g. {"lora_adapter": "runs/qwen25-coder-7b-lora-r16-v1/final",
    #       "git_sha": "abc123"} — anything you want preserved as part
    # of the run record. Goes into wandb.config["metadata"].
    metadata: dict = field(default_factory=dict)


def auto_run_name(cfg: RunConfig) -> str:
    """Compact identifier when wandb_run_name is not set explicitly."""
    model_tag = cfg.llm_model.split("/")[-1][:24]
    return "_".join(
        [
            model_tag,
            cfg.retriever_type[:5],
            f"k{cfg.top_k}",
            f"n{cfg.eval_sample_size}",
        ]
    )
