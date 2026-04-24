"""Shared data models used across API endpoints."""

from pydantic import BaseModel


class SourceReference(BaseModel):
    """Source reference with full metadata."""

    source_id: int
    document_name: str
    chunk_text: str
    relevance_score: float
    folder_path: str | None = None
