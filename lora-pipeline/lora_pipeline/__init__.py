"""Documents → LoRA adapter in one run.

    from pathlib import Path

    from lora_pipeline import PipelineConfig, run_pipeline

    run_pipeline(PipelineConfig(
        docs_dir=Path("my-docs"),
        output_dir=Path("/tmp/rag-lora-smoke"),
        teacher_api_url="http://127.0.0.1:8000/v1",
        teacher_api_key="EMPTY",
        max_chunks=10,
        skip_train=True,
    ))

The current manifest serializes `teacher_api_key`; do not pass a real
cloud credential until secret redaction is implemented.
"""

from lora_pipeline.config import PipelineConfig
from lora_pipeline.pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
