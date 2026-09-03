"""Orchestration: documents in, trained adapter out.

    docs/ ──ingest──▶ ChromaDB ──synth──▶ train.jsonl ──train──▶ adapter/

Each stage writes artifacts into `output_dir`. Ingest and synth can reuse
existing outputs; training runs again unless explicitly skipped.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from lora_pipeline.config import PipelineConfig
from lora_pipeline.steps import preflight_teacher, step_ingest, step_synth, step_train


def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _run_stage(name: str, fn: Callable[[PipelineConfig], dict], cfg: PipelineConfig) -> dict:
    logger.info("─── stage: {name} ───", name=name)
    started = time.monotonic()
    result = fn(cfg)
    result["duration_sec"] = round(time.monotonic() - started, 1)
    logger.info("stage {name} done in {d}s: {r}", name=name, d=result["duration_sec"], r=result)
    return result


def _write_manifest(cfg: PipelineConfig, stages: dict) -> Path:
    contract = cfg.resolved_contract()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_dir": str(cfg.docs_dir),
        "output_dir": str(cfg.output_dir),
        "base_model": cfg.base_model,
        "teacher_model": cfg.teacher_model,
        "contract": contract.to_dict(),
        "stages": stages,
        "config": asdict(cfg),
    }
    with cfg.manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("manifest → {p}", p=cfg.manifest_path)
    return cfg.manifest_path


def run_pipeline(cfg: PipelineConfig) -> dict:
    """Run every non-skipped stage and return their result summaries.

    The complete config, prompt contract and stage summaries are also written
    to ``manifest.json`` in ``output_dir``.
    """
    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    contract = cfg.resolved_contract()
    contract.save(cfg.output_dir)
    logger.info(
        "run → {out} | contract={name} fingerprint={fp} context_chunks={k}",
        out=cfg.output_dir,
        name=contract.name,
        fp=contract.fingerprint(),
        k=contract.context_chunks,
    )

    if cfg.preflight and not cfg.skip_synth:
        preflight_teacher(cfg)

    stages: dict[str, dict] = {}
    if cfg.skip_ingest:
        logger.info("ingest: skipped by config")
    else:
        stages["ingest"] = _run_stage("ingest", step_ingest, cfg)

    if cfg.skip_synth:
        logger.info("synth: skipped by config")
    else:
        stages["synth"] = _run_stage("synth", step_synth, cfg)

    if cfg.skip_train:
        logger.info("train: skipped by config — dataset is ready at {p}", p=cfg.dataset_dir)
    else:
        if not cfg.train_jsonl.exists():
            raise FileNotFoundError(
                f"no training data at {cfg.train_jsonl} — run the synth stage first"
            )
        stages["train"] = _run_stage("train", step_train, cfg)

    _write_manifest(cfg, stages)

    if "train" in stages:
        logger.info("adapter ready → {p}", p=cfg.final_adapter_dir)
    return stages
