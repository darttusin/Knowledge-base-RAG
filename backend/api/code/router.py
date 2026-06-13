from fastapi import APIRouter, Depends

from auth import get_current_user_id
from settings import settings

from api.message.code_parser import execute_code_block

from .models import ErrorMessage, ExecuteCodeRequest, ExecuteCodeResponse

router = APIRouter(tags=["Code"], prefix="/api/code")


@router.post(
    "/execute",
    response_model=ExecuteCodeResponse,
    responses={
        401: {"model": ErrorMessage, "description": "Unauthorized"},
    },
    summary="Выполнить Python-код",
    description=(
        "Исполняет произвольный Python-код в изолированной песочнице "
        "(сервис code-executor). Используется фронтендом для запуска кода "
        "из ответа модели, в том числе отредактированного пользователем."
    ),
)
async def execute_code(
    request: ExecuteCodeRequest,
    _user_id: int = Depends(get_current_user_id),
) -> ExecuteCodeResponse:
    result = await execute_code_block(request.code, settings.CODE_EXECUTOR_URL)
    return ExecuteCodeResponse(**result)
