from pydantic import BaseModel


class ForwardRequest(BaseModel):
    tg_user_id: int | None = None
    text: str


class ForwardResponse(BaseModel):
    response: str
