"""CLI entrypoint.

Usage:
    cd dataset-prep
    uv run python -m dataset_prep \
        --csv ../data/stackoverflow-pytorch.csv \
        --out ../data/sft

The pipeline is RAG-aware: it retrieves PyTorch documentation context
from the same ChromaDB index used in production (default:
data/chromadb, collection docs_fast) and injects ~15% adversarial
refusal examples.

Outputs:
    <out>/train.jsonl
    <out>/val.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from dataset_prep.filtering import FilterConfig
from dataset_prep.pipeline import PipelineConfig, run_pipeline
from dataset_prep.retrieval import RetrievalConfig

MODULE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "stackoverflow-pytorch.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "sft"
DEFAULT_CHROMA = PROJECT_ROOT / "data" / "chromadb"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare RAG-aware SFT dataset from StackOverflow CSV")

    # input / output
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # filters
    p.add_argument("--min-score", type=int, default=5)
    p.add_argument("--min-question-chars", type=int, default=50)
    p.add_argument("--max-question-chars", type=int, default=4000)
    p.add_argument("--min-answer-chars", type=int, default=100)
    p.add_argument("--max-answer-chars", type=int, default=6000)

    # retrieval
    p.add_argument("--chroma-path", type=Path, default=DEFAULT_CHROMA)
    p.add_argument("--collection", type=str, default="docs_fast")
    p.add_argument("--embedding-model", type=str, default="BAAI/bge-base-en-v1.5")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument("--adversarial-fraction", type=float, default=0.15)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.csv.exists():
        logger.error("CSV not found: {path}", path=args.csv)
        sys.exit(1)
    if not args.chroma_path.exists():
        logger.error("ChromaDB not found: {path}", path=args.chroma_path)
        sys.exit(1)

    config = PipelineConfig(
        csv_path=args.csv,
        output_dir=args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        filter_config=FilterConfig(
            min_score=args.min_score,
            min_question_chars=args.min_question_chars,
            max_question_chars=args.max_question_chars,
            min_answer_chars=args.min_answer_chars,
            max_answer_chars=args.max_answer_chars,
        ),
        retrieval_config=RetrievalConfig(
            chroma_path=str(args.chroma_path),
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            device=args.device,
            adversarial_fraction=args.adversarial_fraction,
            seed=args.seed,
        ),
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
