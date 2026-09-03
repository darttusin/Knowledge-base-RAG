"""Convert prepared JSONL into HuggingFace Datasets with chat templates applied.

Input (from `dataset-synth` or `dataset-prep`):
    {
        "question": "...",
        "answer": "...",
        "chunks": [{"id": 1, "source": "...", "text": "..."}],  // preferred
        "context": "...",          // legacy flat string, still accepted
        "is_adversarial": false    // true for synthetic refusal examples
    }

The chat messages are built by the `PromptContract` in `DataConfig`, which
is the same object serving and evaluation use. Storing chunks structurally
rather than pre-rendered means one dataset can be trained under different
contracts without being regenerated.

Output rows fed to SFTTrainer:
    {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...
              <|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"}
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from prompt_contract import PromptContract
from transformers import PreTrainedTokenizerBase

from lora_train.config import DataConfig


def _build_messages(example: dict, contract: PromptContract) -> dict:
    """Render one row into chat messages using the prompt contract.

    Rows from `dataset-synth` carry `chunks` (structured, contract-agnostic);
    older rows from `dataset-prep` carry a flat `context` string. Both are
    rendered through the same contract so training and serving agree.
    """
    chunks = example.get("chunks")
    if chunks:
        context = contract.render_context(chunks)
    else:
        context = str(example.get("context", ""))
    return {
        "messages": contract.build_messages(
            question=example["question"],
            context=context,
            answer=example["answer"],
        )
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
            fn_kwargs={"contract": config.contract},
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
        "formatted datasets ready: train={n_train} val={n_val} contract={c}({fp})",
        n_train=len(processed["train"]),
        n_val=len(processed["validation"]),
        c=config.contract.name,
        fp=config.contract.fingerprint(),
    )
    return processed
