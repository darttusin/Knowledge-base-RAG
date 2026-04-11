from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from rag.models import RetrievedChunk
from rag.vectorstore import search

if TYPE_CHECKING:
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from rag.llm import ChatModel

logger = logging.getLogger(__name__)


def retrieve(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    query: str,
    n_results: int = 5,
) -> list[RetrievedChunk]:
    results = search(collection, embed_model, query, n_results)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    return [
        RetrievedChunk(
            id=i + 1,
            text=doc,
            source=meta.get("source", "N/A"),
            score=1.0 - float(dist),
        )
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists))
    ]


def retrieve_with_rerank(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    query: str,
    n_results: int = 5,
    fetch_k: int = 20,
) -> list[RetrievedChunk]:
    results = search(collection, embed_model, query, n_results=fetch_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return []

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, metas, scores), key=lambda x: x[2], reverse=True)

    return [
        RetrievedChunk(
            id=i + 1,
            text=doc,
            source=meta.get("source", "N/A"),
            score=float(score),
        )
        for i, (doc, meta, score) in enumerate(ranked[:n_results])
    ]


# ---------------------------------------------------------------------------
# Query Transformation (Query Rewriting + HyDE)
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """Rewrite the following user question into a concise, keyword-rich \
search query optimized for retrieving relevant chunks from PyTorch documentation.
- Remove conversational words and filler
- Add relevant technical terms and synonyms
- Output ONLY the rewritten query, nothing else
- Do NOT explain your reasoning

Original question: {query}
Rewritten query:"""

HYDE_PROMPT = """Write a short (3-5 sentences) technical answer to the following \
PyTorch question, as if taken from official PyTorch documentation.
Include relevant API names, class names, and code patterns.
Do NOT add disclaimers or reasoning. Output ONLY the answer.

Question: {query}
Answer:"""


def _llm_with_retry(
    llm: ChatModel,
    prompt: str,
    max_tokens: int = 128,
    max_retries: int = 5,
) -> str:
    for attempt in range(max_retries):
        try:
            return llm.invoke(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            err = str(e).lower()
            if "429" in str(e) or "rate" in err or "overloaded" in err:
                wait = 2**attempt + 1
                logger.warning(
                    "Rate limit, retrying in %ds (attempt %d)", wait, attempt + 1
                )
                time.sleep(wait)
            else:
                raise
    msg = "LLM rate limit: max retries exceeded"
    raise RuntimeError(msg)


def rewrite_query(llm: ChatModel, query: str) -> str:
    return _llm_with_retry(llm, REWRITE_PROMPT.format(query=query), max_tokens=128)


def generate_hypothetical_answer(llm: ChatModel, query: str) -> str:
    return _llm_with_retry(llm, HYDE_PROMPT.format(query=query), max_tokens=256)


def retrieve_with_query_transform(
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    llm: ChatModel,
    query: str,
    n_results: int = 5,
    fetch_k: int = 20,
) -> list[RetrievedChunk]:
    rewritten = rewrite_query(llm, query)
    hyde_answer = generate_hypothetical_answer(llm, query)

    logger.info("[QR] Rewritten: %s", rewritten)
    logger.info("[HyDE] Hypothesis: %s", hyde_answer[:120])

    all_docs: list[str] = []
    all_metas: list[dict] = []
    seen_keys: set[str] = set()

    for q in [query, rewritten, hyde_answer]:
        results = search(collection, embed_model, q, n_results=fetch_k)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            key = doc[:150]
            if key not in seen_keys:
                seen_keys.add(key)
                all_docs.append(doc)
                all_metas.append(meta)

    if not all_docs:
        return []

    pairs = [[query, doc] for doc in all_docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True)

    return [
        RetrievedChunk(
            id=i + 1,
            text=doc,
            source=meta.get("source", "N/A"),
            score=float(score),
        )
        for i, (doc, meta, score) in enumerate(ranked[:n_results])
    ]
