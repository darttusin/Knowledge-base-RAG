import time
from typing import Annotated, Any

from db import get_db
from fastapi import APIRouter, Depends, HTTPException, Request
from schemas.forward import ForwardRequest, ForwardResponse
from services.history_service import save_request
from services.rag_service import chromadb_deduplication_search, rag_answer
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/forward", response_model=ForwardResponse)
async def forward(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    start_time = time.time()
    request_type = "json"
    status_val = "success"
    error_message = None
    response_data: dict[str, Any] | None = None

    try:
        try:
            body = await request.json()
            request_data = ForwardRequest(**body)
        except Exception:
            raise HTTPException(status_code=400, detail="bad request")

        request_data_dict = request_data.model_dump()
        text = request_data.text
        input_length = len(text)
        input_tokens = 0

        try:
            result_text = rag_answer(text, chromadb_deduplication_search(text))
        except Exception as e:
            status_val = "model_failed"
            error_message = str(e)
            raise HTTPException(status_code=403, detail="Model failed")

        processing_time = (time.time() - start_time) * 1000
        response_data = {"response": result_text}
        response = ForwardResponse(response=result_text)

        await save_request(
            session=db,
            request_type=request_type,
            request_data=request_data_dict,
            response_data=response_data,
            processing_time_ms=processing_time,
            input_length=input_length,
            input_tokens=input_tokens,
            image_width=None,
            image_height=None,
            status=status_val,
            error_message=error_message,
            tg_user_id=request_data.tg_user_id,
        )

        return response

    except HTTPException as e:
        if e.status_code == 403:
            raise

        processing_time = (time.time() - start_time) * 1000
        status_val = "error"
        error_message = str(e.detail)

        try:
            body = await request.json()
            request_data_dict = ForwardRequest(**body).model_dump()
            tg_user_id = request_data_dict.get("tg_user_id")
        except:
            request_data_dict = {}
            tg_user_id = None

        await save_request(
            session=db,
            request_type=request_type,
            request_data=request_data_dict,
            response_data=None,
            processing_time_ms=processing_time,
            input_length=None,
            input_tokens=None,
            image_width=None,
            image_height=None,
            status=status_val,
            error_message=error_message,
            tg_user_id=tg_user_id,
        )
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        status_val = "error"
        error_message = str(e)

        try:
            body = await request.json()
            request_data_dict = ForwardRequest(**body).model_dump()
            tg_user_id = request_data_dict.get("tg_user_id")
        except:
            request_data_dict = {}
            tg_user_id = None

        await save_request(
            session=db,
            request_type=request_type,
            request_data=request_data_dict,
            response_data=None,
            processing_time_ms=processing_time,
            input_length=None,
            input_tokens=None,
            image_width=None,
            image_height=None,
            status=status_val,
            error_message=error_message,
            tg_user_id=tg_user_id,
        )

        raise HTTPException(status_code=400, detail="bad request")
