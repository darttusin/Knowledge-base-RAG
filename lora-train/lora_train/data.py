"""Convert prepared JSONL into HuggingFace Datasets with chat templates applied.

Input (from `dataset-prep`, RAG-aware):
    {
        "question": "...",
        "answer": "...",
        "score": 42,
        "context": "...",          // top-k retrieved PyTorch doc chunks
        "is_adversarial": false    // true for synthetic refusal examples
    }

The user turn is constructed as:
    Context:
    {context}

    Question: {question}

So the model is trained to ground its answer in the provided context —
matching the format used at inference time in the production RAG flow.

Output rows fed to SFTTrainer:
    {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...
              <|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"}
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from transformers import PreTrainedTokenizerBase

from lora_train.config import DataConfig


def _build_messages(example: dict, system_prompt: str) -> dict:
    user_content = (
        f"Context:\n{example['context']}\n\n"
        f"Question: {example['question']}"
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
