from datetime import datetime
from typing import Any

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: int
    tg_user_id: int | None
    request_type: str
    request_data: dict[str, Any]
    response_data: dict[str, Any] | None
    processing_time_ms: float | None
    input_length: int | None
    input_tokens: int | None
    image_width: int | None
    image_height: int | None
    status: str
    error_message: str | None
    created_at: datetime
