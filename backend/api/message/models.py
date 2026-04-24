from enum import Enum

from pydantic import BaseModel

from api.shared import SourceReference


class MessageFeedbackEnum(str, Enum):
    like = "like"
    dislike = "dislike"


class SendMessage(BaseModel):
    dialogue_id: int
    message: str
    parent_message_id: int | None = None


class CodeExecution(BaseModel):
    code: str
    success: bool
    stdout: str
    stderr: str
    result: str | None = None
    error: str | None = None


class MessageResponse(BaseModel):
    message_id: int
    parent_message_id: int | None = None
    user_message: str
    assistant_response: str
    sources: list[SourceReference]
    code_executions: list[CodeExecution] = []
    created_at: str


class MessageFeedback(BaseModel):
    message_id: int
    feedback: MessageFeedbackEnum


class ErrorMessage(BaseModel):
    detail: str
