"""Convert prepared JSONL into HuggingFace Datasets with chat templates applied.

Input (from `dataset-prep`):
    {"question": "...", "answer": "...", "score": 42}

Output rows fed to SFTTrainer:
    {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...
              <|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"}

The chat template is applied via the model's tokenizer so the format
matches whatever the base model expects (Qwen2.5 ChatML in our case).
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from lora_train.config import DataConfig


def _build_messages(example: dict, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def _apply_template(example: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_datasets(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """Load train/val JSONL and pre-format each row as a single `text` string.

    SFTTrainer accepts pre-formatted text via `dataset_text_field="text"`.
    """
    if not config.train_jsonl.exists():
        raise FileNotFoundError(f"train jsonl not found: {config.train_jsonl}")
    if not config.val_jsonl.exists():
        raise FileNotFoundError(f"val jsonl not found: {config.val_jsonl}")

    raw = load_dataset(
        "json",
        data_files={
            "train": str(config.train_jsonl),
            "validation": str(config.val_jsonl),
        },
    )
    logger.info(
        "loaded raw datasets: train={n_train} val={n_val}",
        n_train=len(raw["train"]),
        n_val=len(raw["validation"]),
    )

    keep_cols = {"text"}

    def _process(split: Dataset) -> Dataset:
        with_messages = split.map(
            _build_messages,
            fn_kwargs={"system_prompt": config.system_prompt},
            remove_columns=split.column_names,
        )
        with_text = with_messages.map(
            _apply_template,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=[c for c in with_messages.column_names if c not in keep_cols],
        )
        return with_text

    processed = DatasetDict({split: _process(raw[split]) for split in raw})
    logger.info(
        "formatted datasets ready: train={n_train} val={n_val}",
        n_train=len(processed["train"]),
        n_val=len(processed["validation"]),
    )
    return processed
