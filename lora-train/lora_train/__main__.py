"""CLI entrypoint for LoRA training.

Usage on the vast.ai server (after `uv sync`):

    cd lora-train
    uv run python -m lora_train \\
        --train-jsonl ../data/sft/train.jsonl \\
        --val-jsonl ../data/sft/val.jsonl \\
        --output-dir runs/qwen25-coder-7b-r16

Override anything via flags; everything else falls back to dataclass
defaults defined in `config.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from lora_train.config import (
    DataConfig,
    LoraConfig,
    LoraTrainConfig,
    ModelConfig,
    TrainingConfig,
)
from lora_train.train import run_training


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a LoRA adapter for PyTorch RAG")

    # data
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--max-seq-length", type=int, default=4096)

    # model
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument(
        "--qlora",
        action="store_true",
        help="Enable 4-bit quantization (QLoRA). Default: plain bf16 LoRA.",
    )
    p.add_argument("--trust-remote-code", action="store_true")

    # lora
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # training
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "none"],
    )

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.train_jsonl.exists():
        logger.error("train jsonl not found: {p}", p=args.train_jsonl)
        sys.exit(1)
    if not args.val_jsonl.exists():
        logger.error("val jsonl not found: {p}", p=args.val_jsonl)
        sys.exit(1)

    cfg = LoraTrainConfig(
        model=ModelConfig(
            name=args.model_name,
            use_qlora=args.qlora,
            trust_remote_code=args.trust_remote_code,
        ),
        lora=LoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        ),
        data=DataConfig(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            max_seq_length=args.max_seq_length,
        ),
        training=TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            seed=args.seed,
            report_to=args.report_to,
            run_name=args.run_name,
        ),
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
