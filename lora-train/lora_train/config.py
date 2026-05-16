"""Configuration dataclasses for LoRA training.

All hyperparameters live here so the experiment is fully described by
a single object that can be logged to wandb / serialized to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert PyTorch assistant. Answer the user's question using ONLY "
    "the information provided in the Context. If the Context does not contain "
    "enough information to answer the question reliably, say so explicitly "
    "instead of guessing. When showing code, use fenced code blocks with the "
    "`python` language tag."
)

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_qlora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    trust_remote_code: bool = False


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    bias: str = "none"


@dataclass
class DataConfig:
    train_jsonl: Path
    val_jsonl: Path
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_seq_length: int = 4096


@dataclass
class TrainingConfig:
    output_dir: Path
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    report_to: str = "wandb"
    run_name: str | None = None


@dataclass
class LoraTrainConfig:
    """Top-level container — what gets serialized as the experiment record."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(  # type: ignore[call-arg]
        train_jsonl=Path("data/sft/train.jsonl"),
        val_jsonl=Path("data/sft/val.jsonl"),
    ))
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(
        output_dir=Path("lora-train/runs/default"),
    ))
