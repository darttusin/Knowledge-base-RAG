"""End-to-end LoRA training driver.

Composes the pieces: load model+tokenizer → attach LoRA → build datasets
→ configure SFTTrainer → train → save adapter.

NOTE on loss masking: we train on the full sequence (loss on system +
user + assistant tokens), not just the assistant turn. Earlier versions
used TRL's DataCollatorForCompletionOnlyLM but it was removed in TRL
0.19+. Full-sequence loss is suboptimal but still trains correctly —
most of the gradient signal comes from the answer tokens anyway. A
future fix is to use SFTConfig(assistant_only_loss=True) once the
Qwen2.5 chat template is confirmed to ship with `{% generation %}`
markers.

The output `output_dir` will contain the LoRA adapter (not the merged
base model). To deploy in vLLM:

    vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \\
      --enable-lora \\
      --lora-modules pytorch-rag=<output_dir>
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from loguru import logger
from trl import SFTConfig, SFTTrainer

from lora_train.config import LoraTrainConfig, TrainingConfig
from lora_train.data import build_datasets
from lora_train.model import attach_lora, load_model_and_tokenizer


def _build_sft_config(cfg: TrainingConfig, max_seq_length: int) -> SFTConfig:
    return SFTConfig(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def _save_run_config(cfg: LoraTrainConfig) -> None:
    cfg.training.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = cfg.training.output_dir / "run_config.json"

    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"unserializable: {type(obj)}")

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=_default, ensure_ascii=False)
    logger.info("wrote run config → {path}", path=config_path)


def run_training(cfg: LoraTrainConfig) -> None:
    """Run the full LoRA training procedure end-to-end."""
    logger.info("starting lora-train run → {out}", out=cfg.training.output_dir)
    _save_run_config(cfg)

    base_model, tokenizer = load_model_and_tokenizer(cfg.model)
    peft_model = attach_lora(base_model, cfg.lora)

    datasets = build_datasets(cfg.data, tokenizer)
    sft_config = _build_sft_config(cfg.training, cfg.data.max_seq_length)

    trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )

    logger.info("trainer ready, starting training")
    trainer.train()

    final_dir = cfg.training.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    # The adapter is only valid under the prompt format it was trained on —
    # ship that format with the weights so serving can verify the match.
    contract_path = cfg.data.contract.save(final_dir)
    logger.info(
        "training done. adapter saved → {path} (contract {c}={fp})",
        path=final_dir,
        c=cfg.data.contract.name,
        fp=cfg.data.contract.fingerprint(),
    )
    logger.info("prompt contract saved → {p}", p=contract_path)
