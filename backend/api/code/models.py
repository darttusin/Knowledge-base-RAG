from pydantic import BaseModel, Field


class ExecuteCodeRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python-код для исполнения")


class ExecuteCodeResponse(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: str | None = None


class ErrorMessage(BaseModel):
    detail: str
