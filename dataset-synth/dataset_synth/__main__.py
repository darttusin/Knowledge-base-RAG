"""CLI entrypoint for synthetic Q&A generation.

Teacher-agnostic — point --teacher-api-url at OpenAI or a vLLM server.

Local vLLM teacher:
    uv run --locked --package dataset-synth python -m dataset_synth \\
        --teacher-api-url http://localhost:18000/v1 \\
        --teacher-api-key EMPTY \\
        --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ \\
        --out /tmp/rag-sft-synth

Small smoke on 10 chunks to inspect quality and limit cost/output:
    uv run --locked --package dataset-synth python -m dataset_synth ... \\
        --max-chunks 10

This still calls the configured endpoint and writes a dataset; it is not an
offline dry run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from dataset_synth.config import SynthConfig
from dataset_synth.pipeline import run_synth

MODULE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODULE_ROOT.parent
DEFAULT_CHROMA = PROJECT_ROOT / "data" / "chromadb"
DEFAULT_OUT = PROJECT_ROOT / "data" / "sft_synth"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate teacher-produced synthetic Q&A from document chunks"
    )

    # chunks
    p.add_argument("--chroma-path", type=str, default=str(DEFAULT_CHROMA))
    p.add_argument("--collection", type=str, default="docs_fast")
    p.add_argument("--min-chunk-chars", type=int, default=300)
    p.add_argument("--max-chunk-chars", type=int, default=4000)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="0 = all; a small value limits teacher calls and output",
    )

    # teacher
    p.add_argument("--teacher-api-url", type=str, required=True)
    p.add_argument("--teacher-api-key", type=str, default="EMPTY")
    p.add_argument("--teacher-model", type=str, default="gpt-4o-mini")
    p.add_argument("--teacher-temperature", type=float, default=0.7)
    p.add_argument("--teacher-max-tokens", type=int, default=1200)
    p.add_argument("--n-qa-per-chunk", type=int, default=3)
    p.add_argument("--max-workers", type=int, default=8)

    # context assembly
    p.add_argument(
        "--context-chunks",
        type=int,
        default=1,
        help="chunks per training example; set to the serving top_k",
    )
    p.add_argument("--near-distractor-fraction", type=float, default=0.7)

    # adversarial + mix
    p.add_argument("--adversarial-fraction", type=float, default=0.20)
    p.add_argument("--mix-jsonl", type=str, default="", help="prepared SO jsonl to blend in")
    p.add_argument("--mix-fraction", type=float, default=0.0, help="SO rows as fraction of synth count")

    # output
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not Path(args.chroma_path).exists():
        logger.error("chroma path not found: {p}", p=args.chroma_path)
        sys.exit(1)

    cfg = SynthConfig(
        chroma_path=args.chroma_path,
        collection_name=args.collection,
        min_chunk_chars=args.min_chunk_chars,
        max_chunk_chars=args.max_chunk_chars,
        max_chunks=args.max_chunks,
        teacher_model=args.teacher_model,
        teacher_api_url=args.teacher_api_url,
        teacher_api_key=args.teacher_api_key,
        teacher_temperature=args.teacher_temperature,
        teacher_max_tokens=args.teacher_max_tokens,
        n_qa_per_chunk=args.n_qa_per_chunk,
        max_workers=args.max_workers,
        context_chunks=args.context_chunks,
        near_distractor_fraction=args.near_distractor_fraction,
        adversarial_fraction=args.adversarial_fraction,
        mix_jsonl=args.mix_jsonl,
        mix_fraction=args.mix_fraction,
        output_dir=args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    run_synth(cfg)


if __name__ == "__main__":
    main()
