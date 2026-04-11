import glob
import hashlib
import logging
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

logger = logging.getLogger(__name__)

MARKDOWN_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]


def hash_file(filepath: str) -> str:
    hasher = hashlib.md5()  # noqa: S324
    with open(filepath, "rb") as f:
        buf = f.read(8192)
        while buf:
            hasher.update(buf)
            buf = f.read(8192)
    return hasher.hexdigest()


def load_documents(dataset_path: str) -> list[Document]:
    md_files = glob.glob(os.path.join(dataset_path, "**/*.md"), recursive=True)
    documents: list[Document] = []
    seen_hashes: set[str] = set()

    logger.info("Found %d .md files", len(md_files))

    for file_path in tqdm(md_files, desc="Loading documents"):
        try:
            file_hash = hash_file(file_path)
            if file_hash in seen_hashes:
                continue
            seen_hashes.add(file_hash)

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                        "folder": os.path.dirname(
                            os.path.relpath(file_path, dataset_path)
                        ),
                        "file_type": "markdown",
                    },
                )
            )
        except Exception:
            logger.exception("Error loading %s", file_path)

    logger.info("Loaded %d unique documents", len(documents))
    return documents


def deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []

    for chunk in chunks:
        preview = chunk.page_content[:100].strip()
        length = len(chunk.page_content)
        signature = f"{preview}_{length}"

        if signature not in seen:
            seen.add(signature)
            unique.append(chunk)

    logger.info("Deduplication: %d -> %d chunks", len(chunks), len(unique))
    return unique


def split_documents(
    documents: list[Document],
    chunk_size: int = 1600,
    chunk_overlap: int = 100,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MARKDOWN_SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return deduplicate_chunks(chunks)
