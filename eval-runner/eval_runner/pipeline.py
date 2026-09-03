"""Build the RAG pipeline (retriever + generator) from a RunConfig.

Returns two callables:
- `rag_fn(question) -> (answer, contexts)` — full retrieve+generate
- `pure_fn(question) -> answer` — same LLM but no context (baseline)

All heavy objects (embed model, reranker, ChromaDB client, LLM client)
are constructed once here and captured in closures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chromadb
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.chains import answer, answer_without_context
from rag.llm import ChatModel
from rag.models import RetrievedChunk
from rag.retriever import (
    retrieve,
    retrieve_with_query_transform,
    retrieve_with_rerank,
)

from eval_runner.config import RunConfig

RagFn = Callable[[str], tuple[str, list[str]]]
PureFn = Callable[[str], str]


@dataclass
class Pipeline:
    rag_fn: RagFn
    pure_fn: PureFn
    embed_model: SentenceTransformer  # retriever model, exposed for inspection/reuse


def _resolve_device(spec: str) -> str:
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


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Construct the full retrieve-augment-generate stack from config."""
    if not cfg.llm_api_url:
        raise ValueError("llm_api_url is required (point at your vLLM endpoint)")

    device = _resolve_device(cfg.device)
    logger.info(
        "loading embedding model {m} on {d}", m=cfg.embedding_model, d=device
    )
    embed = SentenceTransformer(cfg.embedding_model, device=device)

    logger.info("opening chromadb at {p} (collection={c})",
                p=cfg.chroma_path, c=cfg.chroma_collection)
    client = chromadb.PersistentClient(path=cfg.chroma_path)
    collection = client.get_collection(cfg.chroma_collection)
    logger.info("collection size: {n} chunks", n=collection.count())

    reranker: CrossEncoder | None = None
    if cfg.retriever_type in ("rerank", "query_transform"):
        logger.info("loading reranker {m} on {d}", m=cfg.rerank_model, d=device)
        reranker = CrossEncoder(cfg.rerank_model, device=device)

    logger.info(
        "configuring LLM model={m} url={u} temp={t}",
        m=cfg.llm_model,
        u=cfg.llm_api_url,
        t=cfg.llm_temperature,
    )
    llm = ChatModel(
        model_name=cfg.llm_model,
        api_url=cfg.llm_api_url,
        api_key=cfg.llm_api_key,
        temperature=cfg.llm_temperature,
        max_output_tokens=cfg.llm_max_tokens,
        timeout=cfg.llm_timeout,
    )

    def _get_chunks(question: str) -> list[RetrievedChunk]:
        if cfg.retriever_type == "vanilla":
            return retrieve(collection, embed, question, n_results=cfg.top_k)
        if cfg.retriever_type == "rerank":
            return retrieve_with_rerank(
                collection,
                embed,
                reranker,
                question,
                n_results=cfg.top_k,
                fetch_k=cfg.fetch_k,
            )
        if cfg.retriever_type == "query_transform":
            return retrieve_with_query_transform(
                collection,
                embed,
                reranker,
                llm,
                question,
                n_results=cfg.top_k,
                fetch_k=cfg.fetch_k,
            )
        raise ValueError(f"unknown retriever_type: {cfg.retriever_type}")

    def rag_fn(question: str) -> tuple[str, list[str]]:
        chunks = _get_chunks(question)
        ans = answer(llm, question, chunks)
        return ans, [c.text for c in chunks]

    def pure_fn(question: str) -> str:
        return answer_without_context(llm, question)

    return Pipeline(rag_fn=rag_fn, pure_fn=pure_fn, embed_model=embed)
