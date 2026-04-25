from datetime import datetime

from pydantic import BaseModel


class FolderCreateRequest(BaseModel):
    name: str
    path: str
    parent_id: int | None = None


class FolderResponse(BaseModel):
    id: int
    name: str
    path: str
    parent_id: int | None
    created_at: datetime
    document_count: int = 0
