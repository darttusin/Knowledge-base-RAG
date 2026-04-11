from enum import Enum

from pydantic import BaseModel


class IconsEnum(str, Enum):
    database = "database"
    doc = "doc"
    browser = "browser"


class PreGeneratedQuery(BaseModel):
    query: str
    icon: IconsEnum


class CreateDialogue(BaseModel):
    name: str = "New conversation"


class UpdateDialogue(BaseModel):
    name: str | None = None


class SourceReference(BaseModel):
    """Source reference with full metadata."""

    source_id: int
    document_name: str
    chunk_text: str
    relevance_score: float
    folder_path: str | None = None


class MessageResponse(BaseModel):
    message_id: int
    user_message: str
    assistant_response: str | None
    sources: list[SourceReference] | None = None
    feedback: str | None
    created_at: str


class DialogueResponse(BaseModel):
    dialogue_id: int
    name: str
    created_at: str
    updated_at: str
    pre_generated_queries: list[PreGeneratedQuery]
    messages: list[MessageResponse]


class ShortDialogue(BaseModel):
    dialogue_id: int
    name: str
    created_at: str
    updated_at: str


class ErrorMessage(BaseModel):
    detail: str
