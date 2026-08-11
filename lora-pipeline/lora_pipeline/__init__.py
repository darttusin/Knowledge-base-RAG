"""Documents → LoRA adapter in one run.

    from lora_pipeline import PipelineConfig, run_pipeline

    run_pipeline(PipelineConfig(
        docs_dir=Path("my-docs"),
        output_dir=Path("runs/my-lora"),
        teacher_api_url="https://api.openai.com/v1",
        teacher_api_key="sk-...",
    ))
"""

from lora_pipeline.config import PipelineConfig
from lora_pipeline.pipeline import run_pipeline

__all__ = ["PipelineConfig", "run_pipeline"]
