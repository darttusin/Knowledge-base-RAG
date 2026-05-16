"""Retrieve PyTorch documentation context for each Q&A pair.

Loads the same ChromaDB index used by the production RAG module
(`data/chromadb/`, collection `docs_fast`) and attaches top-k retrieved
chunks to each pair as `context`. This converts plain SFT data
(question → answer) into RAG-aware SFT data (context + question → answer),
eliminating the train/inference mismatch.

Also produces adversarial examples: ~15% of pairs get unrelated context
with a "cannot answer" target answer — this teaches the model to refuse
when the context lacks the answer instead of hallucinating from memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chromadb
from loguru import logger
from sentence_transformers import SentenceTransformer

from dataset_prep.filtering import Pair

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

ADVERSARIAL_ANSWERS: tuple[str, ...] = (
    "Based on the provided context, I cannot answer this question — it does not "
    "contain the relevant information.",
    "The provided context does not contain information to answer this question.",
    "I don't have enough information in the provided context to answer this "
    "question reliably.",
)


@dataclass(frozen=True)
class RetrievalConfig:
    chroma_path: str = "data/chromadb"
    collection_name: str = "docs_fast"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    top_k: int = 5
    device: str = "auto"
    adversarial_fraction: float = 0.15
    seed: int = 42


@dataclass
class RetrievalContext:
    """Loaded chromadb collection + embedding model. Construct once, reuse."""

    collection: Collection
    embed_model: SentenceTransformer
    config: RetrievalConfig


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


def load_retrieval_context(config: RetrievalConfig) -> RetrievalContext:
    """Load the ChromaDB collection and embedding model. Heavy — do once."""
    device = _resolve_device(config.device)
    logger.info(
        "loading embedding model {m} on {d}",
        m=config.embedding_model,
        d=device,
    )
    embed_model = SentenceTransformer(config.embedding_model, device=device)

    client = chromadb.PersistentClient(path=config.chroma_path)
    collection = client.get_collection(config.collection_name)
    logger.info(
        "loaded collection {n}: {c} chunks",
        n=config.collection_name,
        c=collection.count(),
    )

    return RetrievalContext(collection=collection, embed_model=embed_model, config=config)


def _format_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(c.strip() for c in chunks)


def enrich_with_context(pairs: list[Pair], ctx: RetrievalContext) -> list[Pair]:
    """Retrieve top-k chunks for each question and attach them as `context`."""
    questions = [p.question for p in pairs]
    logger.info("embedding {n} questions in batch", n=len(questions))

    query_embeddings = ctx.embed_model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    logger.info("retrieving top-{k} chunks per question", k=ctx.config.top_k)
    enriched: list[Pair] = []
    for pair, embedding in zip(pairs, query_embeddings, strict=True):
        results = ctx.collection.query(
            query_embeddings=[embedding],
            n_results=ctx.config.top_k,
            include=["documents"],
        )
        chunks = results["documents"][0]
        enriched.append(Pair(
            question=pair.question,
            answer=pair.answer,
            score=pair.score,
            context=_format_context(chunks),
            is_adversarial=False,
        ))
    return enriched


def add_adversarial_examples(pairs: list[Pair], ctx: RetrievalContext) -> list[Pair]:
    """Append synthetic refusal examples with unrelated context.

    For ~adversarial_fraction of the existing pairs we create a new pair where:
    - the question is reused (so the natural question distribution is preserved),
    - the context is replaced with `top_k` random chunks from the index,
    - the answer is a refusal phrase ("cannot answer from this context").

    The combined list is shuffled so adversarial examples aren't clustered.
    """
    rng = random.Random(ctx.config.seed)
    n_adversarial = int(len(pairs) * ctx.config.adversarial_fraction)
    if n_adversarial <= 0:
        logger.info(
            "skipping adversarial step (fraction={f})",
            f=ctx.config.adversarial_fraction,
        )
        return pairs

    all_chunks = ctx.collection.get(include=["documents"])
    all_docs: list[str] = all_chunks["documents"]
    if len(all_docs) < ctx.config.top_k:
        raise ValueError(
            f"collection has only {len(all_docs)} chunks, need at least {ctx.config.top_k}"
        )

    sampled = rng.sample(pairs, n_adversarial)
    adversarial: list[Pair] = []
    for pair in sampled:
        random_chunks = rng.sample(all_docs, ctx.config.top_k)  # noqa: S311
        adversarial.append(Pair(
            question=pair.question,
            answer=rng.choice(ADVERSARIAL_ANSWERS),  # noqa: S311
            score=pair.score,
            context=_format_context(random_chunks),
            is_adversarial=True,
        ))

    combined = pairs + adversarial
    rng.shuffle(combined)
    logger.info(
        "added {n} adversarial examples ({pct:.1f}%); total={t}",
        n=n_adversarial,
        pct=100 * n_adversarial / len(combined),
        t=len(combined),
    )
    return combined


def enrich_and_augment(pairs: list[Pair], ctx: RetrievalContext) -> list[Pair]:
    """Convenience wrapper: enrich with context, then add adversarial examples."""
    enriched = enrich_with_context(pairs, ctx)
    return add_adversarial_examples(enriched, ctx)
