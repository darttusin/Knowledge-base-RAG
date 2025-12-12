from datetime import datetime
from typing import Any

from db import RequestHistory
from schemas.history import HistoryItem
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession


async def save_request(
    session: AsyncSession,
    request_type: str,
    request_data: dict[str, Any],
    response_data: dict[str, Any] | None,
    processing_time_ms: float | None,
    input_length: int | None,
    input_tokens: int | None,
    image_width: int | None,
    image_height: int | None,
    status: str,
    error_message: str | None,
    tg_user_id: int | None = None,
) -> RequestHistory:
    history_item = RequestHistory(
        tg_user_id=tg_user_id,
        request_type=request_type,
        request_data=request_data,
        response_data=response_data,
        processing_time_ms=processing_time_ms,
        input_length=input_length,
        input_tokens=input_tokens,
        image_width=image_width,
        image_height=image_height,
        status=status,
        error_message=error_message,
        created_at=datetime.now(),
    )
    session.add(history_item)
    await session.commit()
    await session.refresh(history_item)
    return history_item


async def get_all_history(
    session: AsyncSession, tg_user_id: int | None = None
) -> list[HistoryItem]:
    stmt = select(RequestHistory)
    if tg_user_id is not None:
        stmt = stmt.where(RequestHistory.tg_user_id == tg_user_id)
    stmt = stmt.order_by(RequestHistory.created_at.desc())
    result = await session.execute(stmt)
    history_records = result.scalars().all()

    return [
        HistoryItem(
            id=record.id,
            tg_user_id=record.tg_user_id,
            request_type=record.request_type,
            request_data=record.request_data,
            response_data=record.response_data,
            processing_time_ms=record.processing_time_ms,
            input_length=record.input_length,
            input_tokens=record.input_tokens,
            image_width=record.image_width,
            image_height=record.image_height,
            status=record.status,
            error_message=record.error_message,
            created_at=record.created_at,
        )
        for record in history_records
    ]


async def delete_all_history(session: AsyncSession) -> int:
    stmt = delete(RequestHistory)
    result = await session.execute(stmt)
    return result.rowcount or 0  # type: ignore
