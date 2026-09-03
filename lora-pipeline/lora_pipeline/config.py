"""One config object describing a full documents → adapter run.

Everything a run needs lives here so the whole experiment is reproducible
from a single serialized object, and so `manifest.json` in the output
directory fully explains how an adapter was produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from prompt_contract import GROUNDED_CONTRACT, PromptContract


@dataclass
class PipelineConfig:
    """Inputs and knobs for `run_pipeline`.

    Only `docs_dir` and `teacher_api_url` are genuinely required; every
    other field has a default that produces a sane small run.
    """

    # === input / output ===
    docs_dir: Path
    output_dir: Path
    extensions: tuple[str, ...] = ("md",)

    # === 1. ingest ===
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    device: str = "auto"
    collection_name: str = "docs"

    # === 2. synthetic dataset ===
    teacher_api_url: str = ""
    teacher_api_key: str = "EMPTY"
    teacher_model: str = "gpt-4o-mini"
    teacher_temperature: float = 0.7
    teacher_max_workers: int = 8
    n_qa_per_chunk: int = 3
    max_chunks: int = 0  # 0 = all; small value for a smoke run
    min_chunk_chars: int = 300
    max_chunk_chars: int = 4000
    adversarial_fraction: float = 0.20
    val_fraction: float = 0.05

    # === 3. training ===
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_qlora: bool = False
    trust_remote_code: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] | str = "all-linear"
    epochs: float = 2.0
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    report_to: str = "none"
    run_name: str | None = None

    # === shared prompt format ===
    #: `context_chunks` is applied to this contract, so the number of chunks
    #: the model trains on and the number it is told to expect cannot drift.
    contract: PromptContract = GROUNDED_CONTRACT
    context_chunks: int = 5

    # === control ===
    skip_ingest: bool = False
    skip_synth: bool = False
    skip_train: bool = False
    force_ingest: bool = False
    force_synth: bool = False
    preflight: bool = True
    seed: int = 42

    # populated at runtime
    metadata: dict = field(default_factory=dict)

    # --- derived paths ---------------------------------------------------

    @property
    def chroma_path(self) -> Path:
        return self.output_dir / "chromadb"

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "dataset"

    @property
    def train_jsonl(self) -> Path:
        return self.dataset_dir / "train.jsonl"

    @property
    def val_jsonl(self) -> Path:
        return self.dataset_dir / "val.jsonl"

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / "adapter"

    @property
    def final_adapter_dir(self) -> Path:
        return self.adapter_dir / "final"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    def resolved_contract(self) -> PromptContract:
        """The contract actually used, with `context_chunks` applied."""
        return self.contract.with_context_chunks(self.context_chunks)

    def validate(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"docs dir not found: {self.docs_dir}")
        if not self.docs_dir.is_dir():
            raise NotADirectoryError(f"docs dir is not a directory: {self.docs_dir}")
        if not self.skip_synth and not self.teacher_api_url:
            raise ValueError(
                "teacher_api_url is required to generate a dataset "
                "(or pass skip_synth with an existing dataset)"
            )
        if self.context_chunks < 1:
            raise ValueError("context_chunks must be >= 1")
