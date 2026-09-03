"""The three stages, each independently skippable and resumable.

Heavy dependencies are imported inside the functions on purpose: producing
a dataset on a laptop should not require the CUDA training stack, and
training on a GPU box should not require the embedding stack to be loaded
before it is needed.
"""

from __future__ import annotations

from loguru import logger

from lora_pipeline.config import PipelineConfig


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def preflight_teacher(cfg: PipelineConfig) -> None:
    """Fail before the expensive stages if the teacher endpoint is unusable.

    Embedding a large corpus takes minutes to hours; discovering a bad URL
    or key after that is a waste of the run.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url=cfg.teacher_api_url,
        api_key=cfg.teacher_api_key,
        timeout=30.0,
    )
    try:
        models = [m.id for m in client.models.list().data]
    except Exception as exc:  # noqa: BLE001 - surfaced with context below
        raise RuntimeError(
            f"teacher endpoint unreachable at {cfg.teacher_api_url}: {exc}"
        ) from exc

    if models and cfg.teacher_model not in models:
        logger.warning(
            "teacher model {m} not in endpoint's model list ({avail}) — "
            "continuing, some gateways do not list every model",
            m=cfg.teacher_model,
            avail=", ".join(models[:5]),
        )
    logger.info("preflight ok: teacher endpoint reachable")


def step_ingest(cfg: PipelineConfig) -> dict:
    """Documents on disk → embedded chunks in a ChromaDB collection."""
    from rag.documents import load_documents, split_documents
    from rag.vectorstore import create_collection, index_chunks
    from sentence_transformers import SentenceTransformer

    cfg.chroma_path.mkdir(parents=True, exist_ok=True)

    if not cfg.force_ingest:
        # get_or_create rather than list_collections: the latter returns
        # collection objects in some chromadb versions and bare names in others.
        probe = create_collection(str(cfg.chroma_path), cfg.collection_name)
        if probe.count():
            logger.info(
                "ingest: reusing existing collection {c} ({n} chunks) — "
                "pass force_ingest to rebuild",
                c=cfg.collection_name,
                n=probe.count(),
            )
            return {"chunks": probe.count(), "reused": True}

    docs = load_documents(str(cfg.docs_dir), extensions=cfg.extensions)
    if not docs:
        raise RuntimeError(
            f"no documents found under {cfg.docs_dir} "
            f"(looked for: {', '.join(cfg.extensions)})"
        )
    chunks = split_documents(docs, cfg.chunk_size, cfg.chunk_overlap)

    device = resolve_device(cfg.device)
    logger.info("ingest: embedding with {m} on {d}", m=cfg.embedding_model, d=device)
    embedder = SentenceTransformer(cfg.embedding_model, device=device)

    collection = create_collection(
        str(cfg.chroma_path),
        cfg.collection_name,
        recreate=cfg.force_ingest,
    )
    index_chunks(collection, chunks, embedder, batch_size=200)

    return {"documents": len(docs), "chunks": collection.count(), "reused": False}


def step_synth(cfg: PipelineConfig) -> dict:
    """Chunks → grounded Q&A pairs with distractors and refusals."""
    from dataset_synth.config import SynthConfig
    from dataset_synth.pipeline import run_synth

    if cfg.train_jsonl.exists() and not cfg.force_synth:
        logger.info(
            "synth: reusing existing dataset at {p} — pass force_synth to regenerate",
            p=cfg.dataset_dir,
        )
        n_train = sum(1 for _ in cfg.train_jsonl.open(encoding="utf-8"))
        return {"train": n_train, "reused": True}

    synth_cfg = SynthConfig(
        chroma_path=str(cfg.chroma_path),
        collection_name=cfg.collection_name,
        min_chunk_chars=cfg.min_chunk_chars,
        max_chunk_chars=cfg.max_chunk_chars,
        max_chunks=cfg.max_chunks,
        teacher_model=cfg.teacher_model,
        teacher_api_url=cfg.teacher_api_url,
        teacher_api_key=cfg.teacher_api_key,
        teacher_temperature=cfg.teacher_temperature,
        max_workers=cfg.teacher_max_workers,
        n_qa_per_chunk=cfg.n_qa_per_chunk,
        context_chunks=cfg.context_chunks,
        adversarial_fraction=cfg.adversarial_fraction,
        output_dir=str(cfg.dataset_dir),
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    summary = run_synth(synth_cfg)
    summary["reused"] = False
    return summary


def step_train(cfg: PipelineConfig) -> dict:
    """Q&A dataset → LoRA adapter, trained under the pipeline's contract."""
    try:
        from lora_train.config import (
            DataConfig,
            LoraConfig,
            LoraTrainConfig,
            ModelConfig,
            TrainingConfig,
        )
        from lora_train.train import run_training
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "the training stack is not installed — run "
            "`uv sync --package lora-pipeline --extra train` on a CUDA machine, "
            "or pass --skip-train to stop after building the dataset"
        ) from exc

    contract = cfg.resolved_contract()
    train_cfg = LoraTrainConfig(
        model=ModelConfig(
            name=cfg.base_model,
            use_qlora=cfg.use_qlora,
            trust_remote_code=cfg.trust_remote_code,
        ),
        lora=LoraConfig(
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
        ),
        data=DataConfig(
            train_jsonl=cfg.train_jsonl,
            val_jsonl=cfg.val_jsonl,
            contract=contract,
            max_seq_length=cfg.max_seq_length,
        ),
        training=TrainingConfig(
            output_dir=cfg.adapter_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            report_to=cfg.report_to,
            run_name=cfg.run_name,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optim=cfg.optim,
        ),
    )
    run_training(train_cfg)
    return {
        "adapter": str(cfg.final_adapter_dir),
        "contract_fingerprint": contract.fingerprint(),
    }
