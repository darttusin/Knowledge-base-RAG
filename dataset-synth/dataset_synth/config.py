"""Configuration for synthetic Q&A generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SynthConfig:
    # === source chunks ===
    chroma_path: str = "data/chromadb"
    collection_name: str = "docs_fast"
    min_chunk_chars: int = 300  # skip tiny/table-only chunks (~11% of corpus)
    max_chunk_chars: int = 4000
    max_chunks: int = 0  # 0 = all; set small for dry runs

    # === teacher LLM (any OpenAI-compatible endpoint) ===
    teacher_model: str = "gpt-4o-mini"
    teacher_api_url: str = ""  # OpenAI: https://api.openai.com/v1 ; vLLM: http://host:port/v1
    teacher_api_key: str = "EMPTY"
    teacher_temperature: float = 0.7  # some diversity in generated questions
    teacher_max_tokens: int = 1200
    teacher_timeout: float = 90.0
    n_qa_per_chunk: int = 3
    max_workers: int = 8  # concurrent teacher requests

    # === adversarial refusal examples ===
    adversarial_fraction: float = 0.20  # of generated pairs

    # === optional mix with prepared SO data (hybrid dataset) ===
    mix_jsonl: str = ""  # path to dataset-prep output (e.g. data/sft/train.jsonl)
    mix_fraction: float = 0.0  # how many SO rows to add, as a fraction of synth count

    # === output ===
    output_dir: str = "data/sft_synth"
    val_fraction: float = 0.05
    seed: int = 42
