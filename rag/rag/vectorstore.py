import logging

import chromadb
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_collection(
    chroma_path: str,
    collection_name: str = "docs_fast",
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    collection: chromadb.Collection,
    chunks: list[Document],
    model: SentenceTransformer,
    batch_size: int = 200,
) -> None:
    if collection.count():
        logger.info(
            "Collection already has %d chunks, skipping indexing", collection.count()
        )
        return

    logger.info("Indexing %d chunks...", len(chunks))
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i : i + batch_size]
        documents_batch = [c.page_content for c in batch]
        metadatas_batch = [c.metadata for c in batch]
        ids_batch = [
            f"{i + j}_{hash(c.page_content[:50]) & 0xFFFFFF}"
            for j, c in enumerate(batch)
        ]
        embeddings_batch = model.encode(
            documents_batch, normalize_embeddings=True
        ).tolist()

        collection.add(
            embeddings=embeddings_batch,
            documents=documents_batch,
            metadatas=metadatas_batch,
            ids=ids_batch,
        )

    logger.info("Indexed %d chunks", len(chunks))


SearchResults = dict[str, list[list]]


def search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    query: str,
    n_results: int = 5,
) -> SearchResults:
    results = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n_results * 3,
        include=["documents", "metadatas", "distances"],
    )
    return _filter_duplicates(results, n_results)


def _filter_duplicates(results: SearchResults, n_results: int) -> SearchResults:
    if not results["documents"]:
        return results

    seen: set[str] = set()
    docs: list[str] = []
    metas: list[dict] = []
    dists: list[float] = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = metadata.get("source", "unknown")
        content_key = f"{source}:{doc[:100]}"

        if content_key not in seen and len(docs) < n_results:
            seen.add(content_key)
            docs.append(doc)
            metas.append(metadata)
            dists.append(distance)

    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [dists],
    }
