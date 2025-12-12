from typing import Annotated

from db import get_db
from fastapi import APIRouter, Depends, Header, HTTPException, status
from services.history_service import delete_all_history, get_all_history
from settings import settings
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("/history")
async def get_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    tg_user_id: int | None = None,
):
    history = await get_all_history(db, tg_user_id=tg_user_id)
    return [item.model_dump() for item in history]


@router.delete("/history")
async def delete_history(
    db: Annotated[AsyncSession, Depends(get_db)],
    confirmation_token: str = Header(..., alias="X-Confirmation-Token"),
):
    if confirmation_token != settings.ADMIN_DELETE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid confirmation token"
        )

    deleted_count = await delete_all_history(db)

    return {"message": "History deleted successfully", "deleted_count": deleted_count}
