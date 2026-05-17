"""CLI entrypoint.

Two usage patterns:

  1. Pure CLI (all flags):
        uv run python -m eval_runner \\
            --llm-model "qwen-base" \\
            --llm-api-url http://server:8000/v1 \\
            --judge-api-url https://api.openai.com/v1 \\
            --judge-api-key sk-... \\
            --retriever-type rerank \\
            --top-k 5 \\
            --wandb-run-name "qwen-base-rerank-k5"

  2. JSON preset + overrides:
        uv run python -m eval_runner \\
            --config configs/lora_rerank.json \\
            --wandb-run-name "lora-rerank-debug-attempt-3"

Any CLI flag overrides the corresponding field from --config.
Field names in the JSON file match RunConfig attribute names exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from loguru import logger

from eval_runner.config import RunConfig
from eval_runner.runner import run_evaluation
from eval_runner.tracking import log_to_wandb


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a RAG evaluation and push results to wandb",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON preset. CLI flags override file values.",
    )

    # === retriever ===
    p.add_argument("--retriever-type", choices=["vanilla", "rerank", "query_transform"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--fetch-k", type=int)
    p.add_argument("--embedding-model", type=str)
    p.add_argument("--rerank-model", type=str)
    p.add_argument("--chroma-path", type=str)
    p.add_argument("--chroma-collection", type=str)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])

    # === generator LLM ===
    p.add_argument("--llm-model", type=str, help="Model name OR vLLM --lora-modules alias")
    p.add_argument("--llm-api-url", type=str)
    p.add_argument("--llm-api-key", type=str)
    p.add_argument("--llm-temperature", type=float)
    p.add_argument("--llm-max-tokens", type=int)
    p.add_argument("--llm-timeout", type=float)

    # === judge LLM ===
    # No --judge-api-key flag: vLLM endpoints don't validate it, and for
    # cloud judges (OpenAI etc.) put the key in a --config JSON preset.
    p.add_argument("--judge-model", type=str)
    p.add_argument("--judge-api-url", type=str)

    # === eval data ===
    p.add_argument("--eval-csv-path", type=str)
    p.add_argument("--eval-sample-size", type=int)
    p.add_argument("--eval-min-answer-score", type=int)
    p.add_argument("--eval-seed", type=int)
    p.add_argument("--eval-embedding-model", type=str)

    # === what to compute ===
    p.add_argument("--no-lexical", action="store_true", help="Skip squad_f1 / precision / recall")
    p.add_argument("--no-semantic", action="store_true", help="Skip answer↔answer similarity")
    p.add_argument("--no-ragas", action="store_true", help="Skip RAGAS (faithfulness etc.)")

    # === tracking ===
    p.add_argument("--wandb-project", type=str)
    p.add_argument("--wandb-run-name", type=str)
    p.add_argument("--wandb-tag", action="append", dest="wandb_tags", default=None,
                   help="Repeatable. e.g. --wandb-tag lora --wandb-tag rerank")
    p.add_argument("--wandb-notes", type=str)
    p.add_argument("--wandb-api-key", type=str, default=None,
                   help="If unset, falls back to WANDB_API_KEY env or prior `wandb login`")

    # === metadata ===
    p.add_argument("--description", type=str)
    p.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help='Free-form JSON, e.g. \'{"lora_adapter":"runs/v1/final","git_sha":"abc"}\'',
    )

    return p


def _load_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.error("config file not found: {p}", p=path)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    """Layer: defaults → JSON file → CLI flags (last wins)."""
    overrides: dict[str, Any] = {}
    if args.config is not None:
        overrides.update(_load_json_config(args.config))

    cli_map = {
        "retriever_type": args.retriever_type,
        "top_k": args.top_k,
        "fetch_k": args.fetch_k,
        "embedding_model": args.embedding_model,
        "rerank_model": args.rerank_model,
        "chroma_path": args.chroma_path,
        "chroma_collection": args.chroma_collection,
        "device": args.device,
        "llm_model": args.llm_model,
        "llm_api_url": args.llm_api_url,
        "llm_api_key": args.llm_api_key,
        "llm_temperature": args.llm_temperature,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_timeout": args.llm_timeout,
        "judge_model": args.judge_model,
        "judge_api_url": args.judge_api_url,
        "eval_csv_path": args.eval_csv_path,
        "eval_sample_size": args.eval_sample_size,
        "eval_min_answer_score": args.eval_min_answer_score,
        "eval_seed": args.eval_seed,
        "eval_embedding_model": args.eval_embedding_model,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_tags": args.wandb_tags,
        "wandb_notes": args.wandb_notes,
        "description": args.description,
    }
    for k, v in cli_map.items():
        if v is not None:
            overrides[k] = v

    if args.no_lexical:
        overrides["compute_lexical"] = False
    if args.no_semantic:
        overrides["compute_semantic"] = False
    if args.no_ragas:
        overrides["compute_ragas"] = False

    if args.metadata_json:
        try:
            overrides["metadata"] = json.loads(args.metadata_json)
        except json.JSONDecodeError as e:
            logger.error("--metadata-json invalid: {e}", e=e)
            sys.exit(1)

    valid_fields = {f.name for f in fields(RunConfig)}
    unknown = set(overrides) - valid_fields
    if unknown:
        logger.error("unknown config keys: {u}", u=sorted(unknown))
        sys.exit(1)

    return RunConfig(**overrides)


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _build_run_config(args)

    if not cfg.llm_api_url:
        logger.error("llm_api_url is required (point at your vLLM endpoint)")
        sys.exit(1)
    if cfg.compute_ragas and not cfg.judge_api_url:
        logger.warning(
            "compute_ragas=True but judge_api_url empty — "
            "ragas step will no-op. Pass --no-ragas to silence."
        )

    result = run_evaluation(cfg)
    url = log_to_wandb(result, api_key=args.wandb_api_key)
    logger.info("done. wandb run: {u}", u=url)


if __name__ == "__main__":
    main()
