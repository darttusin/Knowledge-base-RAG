"""One command: a folder of documentation in, a trained LoRA adapter out.

Smallest useful invocation — generate the dataset with an OpenAI teacher
and train on the local GPU:

    uv run python -m lora_pipeline \\
        --docs-dir ./my-docs \\
        --output-dir runs/my-lora \\
        --teacher-api-url https://api.openai.com/v1 \\
        --teacher-api-key sk-... \\
        --teacher-model gpt-4o-mini

Same thing with a local teacher served by vLLM (no API costs):

    uv run python -m lora_pipeline \\
        --docs-dir ./my-docs \\
        --output-dir runs/my-lora \\
        --teacher-api-url http://localhost:8000/v1 \\
        --teacher-model Qwen/Qwen2.5-32B-Instruct-AWQ

Check the shape of the data before paying for a full generation pass:

    ... --max-chunks 20 --skip-train

Produce the dataset on a laptop, train later on a GPU box:

    ... --skip-train                    # laptop
    ... --skip-ingest --skip-synth      # GPU box, same --output-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from prompt_contract import get_contract

from lora_pipeline.config import PipelineConfig
from lora_pipeline.pipeline import run_pipeline


def _add_io_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("input / output")
    g.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="folder with your documentation (searched recursively)",
    )
    g.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="everything this run produces lands here: index, dataset, adapter",
    )
    g.add_argument(
        "--ext",
        type=str,
        default="md",
        help="comma-separated file extensions to ingest (default: md)",
    )


def _add_ingest_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("1. ingest")
    g.add_argument("--chunk-size", type=int, default=1000)
    g.add_argument("--chunk-overlap", type=int, default=200)
    g.add_argument("--embedding-model", type=str, default="BAAI/bge-base-en-v1.5")
    g.add_argument("--collection", type=str, default="docs")
    g.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])


def _add_synth_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("2. dataset generation")
    g.add_argument("--teacher-api-url", type=str, default="")
    g.add_argument("--teacher-api-key", type=str, default="EMPTY")
    g.add_argument("--teacher-model", type=str, default="gpt-4o-mini")
    g.add_argument("--teacher-temperature", type=float, default=0.7)
    g.add_argument("--teacher-workers", type=int, default=8)
    g.add_argument("--qa-per-chunk", type=int, default=3)
    g.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="0 = use every chunk; a small number gives a cheap smoke run",
    )
    g.add_argument("--adversarial-fraction", type=float, default=0.20)
    g.add_argument("--val-fraction", type=float, default=0.05)
    g.add_argument(
        "--context-chunks",
        type=int,
        default=5,
        help="chunks per training example; match your retriever's top_k",
    )
    g.add_argument(
        "--contract",
        type=str,
        default="grounded",
        help="prompt format: built-in name (grounded, sourced) or a JSON path",
    )


def _add_train_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("3. training")
    g.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    g.add_argument("--qlora", action="store_true", help="4-bit base weights (less VRAM)")
    g.add_argument("--trust-remote-code", action="store_true")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--lora-targets", type=str, default="all-linear")
    g.add_argument("--epochs", type=float, default=2.0)
    g.add_argument("--batch-size", type=int, default=1)
    g.add_argument("--grad-accum", type=int, default=16)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--max-seq-length", type=int, default=4096)
    g.add_argument("--optim", type=str, default="paged_adamw_8bit")
    g.add_argument("--no-gradient-checkpointing", dest="grad_ckpt", action="store_false")
    g.add_argument("--report-to", type=str, default="none", choices=["wandb", "tensorboard", "none"])
    g.add_argument("--run-name", type=str, default=None)


def _add_control_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("control")
    g.add_argument("--skip-ingest", action="store_true")
    g.add_argument("--skip-synth", action="store_true")
    g.add_argument("--skip-train", action="store_true", help="stop after building the dataset")
    g.add_argument("--force-ingest", action="store_true", help="rebuild the index from scratch")
    g.add_argument("--force-synth", action="store_true", help="regenerate the dataset")
    g.add_argument("--no-preflight", dest="preflight", action="store_false")
    g.add_argument("--seed", type=int, default=42)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lora_pipeline",
        description="Documentation folder → grounded dataset → trained LoRA adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_io_args(p)
    _add_ingest_args(p)
    _add_synth_args(p)
    _add_train_args(p)
    _add_control_args(p)
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    targets = args.lora_targets
    if targets != "all-linear":
        targets = tuple(t.strip() for t in targets.split(",") if t.strip())

    return PipelineConfig(
        docs_dir=args.docs_dir,
        output_dir=args.output_dir,
        extensions=tuple(e.strip() for e in args.ext.split(",") if e.strip()),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        device=args.device,
        teacher_api_url=args.teacher_api_url,
        teacher_api_key=args.teacher_api_key,
        teacher_model=args.teacher_model,
        teacher_temperature=args.teacher_temperature,
        teacher_max_workers=args.teacher_workers,
        n_qa_per_chunk=args.qa_per_chunk,
        max_chunks=args.max_chunks,
        adversarial_fraction=args.adversarial_fraction,
        val_fraction=args.val_fraction,
        base_model=args.base_model,
        use_qlora=args.qlora,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.grad_ckpt,
        optim=args.optim,
        report_to=args.report_to,
        run_name=args.run_name,
        contract=get_contract(args.contract),
        context_chunks=args.context_chunks,
        skip_ingest=args.skip_ingest,
        skip_synth=args.skip_synth,
        skip_train=args.skip_train,
        force_ingest=args.force_ingest,
        force_synth=args.force_synth,
        preflight=args.preflight,
        seed=args.seed,
    )


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cfg = _config_from_args(args)
        run_pipeline(cfg)
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        logger.error("{e}", e=exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
