from typing import Annotated

from auth import get_current_admin
from db import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from services.stats_service import get_stats as get_stats_service
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
security = HTTPBearer(auto_error=False)


@router.get("/stats")
async def get_stats(
    db: Annotated[AsyncSession, Depends(get_db)],
    authorization: HTTPAuthorizationCredentials | None = Depends(security),
    tg_user_id: int | None = None,
):
    is_admin = False

    if authorization:
        try:
            token = authorization.credentials
            _ = get_current_admin(token)
            is_admin = True
        except HTTPException:
            pass

    if tg_user_id is None and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="tg_user_id is required when not authenticated as admin",
        )

    stats = await get_stats_service(db, tg_user_id=tg_user_id)
    return stats
