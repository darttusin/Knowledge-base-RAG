import re
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from settings import settings

CHROMA_CLIENT = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
COLLECTION: chromadb.Collection = CHROMA_CLIENT.get_or_create_collection(
    name=settings.CHROMADB_COLLECTION, metadata={"hnsw:space": "cosine"}
)
MODEL: SentenceTransformer = SentenceTransformer(settings.MODEL_NAME)


def get_best_preview(content, length=150):
    code_match = re.search(r"```.*?```", content, re.DOTALL)
    if code_match:
        code_text = code_match.group(0)
        if len(code_text) > length:
            return code_text[:length] + "..."
        return code_text

    if len(content) > length:
        return content[:length] + "..."
    return content


def chromadb_deduplication_search(
    query: str, n_results: int = 5
) -> dict[str, list[Any]]:
    initial_results = COLLECTION.query(
        query_embeddings=MODEL.encode([query]).tolist(),
        n_results=n_results * 3,
        include=["documents", "metadatas", "distances"],
    )

    if not initial_results["documents"]:
        return initial_results  # type: ignore

    seen_sources = set()
    unique_docs = []
    unique_metadatas = []
    unique_distances = []

    for doc, metadata, distance in zip(
        initial_results["documents"][0],
        initial_results["metadatas"][0]
        if initial_results["metadatas"]
        else [{}] * len(initial_results["documents"]),
        initial_results["distances"][0]
        if initial_results["distances"]
        else [0] * len(initial_results["documents"]),
    ):
        source = metadata.get("source", "unknown")
        preview = doc[:100]
        content_key = f"{source}:{preview}"

        if content_key not in seen_sources and len(unique_docs) < n_results:
            seen_sources.add(content_key)
            unique_docs.append(doc)
            unique_metadatas.append(metadata)
            unique_distances.append(distance)

    return {
        "documents": [unique_docs],
        "metadatas": [unique_metadatas],
        "distances": [unique_distances],
    }
