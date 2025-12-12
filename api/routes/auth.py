from auth import create_access_token
from fastapi import APIRouter, HTTPException, status
from settings import settings

router = APIRouter()


@router.post("/api/auth/login")
async def login(username: str, password: str):
    if username == settings.ADMIN_LOGIN and password == settings.ADMIN_PASSWORD:
        access_token = create_access_token(data={"sub": username, "role": "admin"})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
    )
