"""CLI entrypoint.

Usage:
    cd dataset-prep
    uv run python -m dataset_prep \
        --csv ../data/stackoverflow-pytorch.csv \
        --out ../data/sft

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

MODULE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "stackoverflow-pytorch.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "sft"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare SFT dataset from StackOverflow CSV")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for JSONL files")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-score", type=int, default=5)
    p.add_argument("--min-question-chars", type=int, default=50)
    p.add_argument("--max-question-chars", type=int, default=4000)
    p.add_argument("--min-answer-chars", type=int, default=100)
    p.add_argument("--max-answer-chars", type=int, default=6000)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.csv.exists():
        logger.error("CSV not found: {path}", path=args.csv)
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
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
