import httpx
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Dialogue, Message
from settings import settings
from services.rag_service import get_rag_service

from .code_parser import parse_and_execute_code
from .models import CodeExecution, MessageFeedback, MessageResponse, SendMessage

# Fallback to external RAG API if local RAG is not available
RAG_API_URL = "http://localhost:8000/forward"


async def send_message(
    data: SendMessage, user_id: int, db: AsyncSession
) -> MessageResponse:
    result = await db.execute(
        select(Dialogue).where(
            Dialogue.id == data.dialogue_id, Dialogue.user_id == user_id
        )
    )
    dialogue = result.scalar_one_or_none()

    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )

    new_message = Message(
        dialogue_id=data.dialogue_id, user_message=data.message, assistant_response=None
    )

    db.add(new_message)
    await db.flush()

    # Try to use local RAG service first
    assistant_response = ""
    sources = []

    if settings.RAG_ENABLED:
        try:
            rag_service = get_rag_service()

            # Answer question using local RAG
            rag_response = rag_service.answer_question(
                question=data.message,
                strategy="rerank",  # Use rerank for balance between speed and quality
                check_topic=settings.OUTLIER_DETECTION_ENABLED,
                reject_off_topic=settings.OUTLIER_REJECT_OFF_TOPIC,
            )

            assistant_response = rag_response.answer

            # Extract sources from chunks
            sources = [chunk.source for chunk in rag_response.chunks]

        except RuntimeError:
            # RAG service not initialized, fallback to external API
            print("⚠ RAG service not available, falling back to external API")
            assistant_response, sources = await _call_external_rag_api(data.message, user_id)

        except Exception as e:
            # Any other error, log and fallback
            print(f"⚠ RAG service error: {e}, falling back to external API")
            assistant_response, sources = await _call_external_rag_api(data.message, user_id)

    else:
        # RAG disabled, use external API
        assistant_response, sources = await _call_external_rag_api(data.message, user_id)

    new_message.assistant_response = assistant_response

    await db.commit()
    await db.refresh(new_message)

    code_results = await parse_and_execute_code(assistant_response, settings.CODE_EXECUTOR_URL)
    code_executions = [CodeExecution(**result) for result in code_results]

    return MessageResponse(
        message_id=new_message.id,
        user_message=new_message.user_message,
        assistant_response=assistant_response,  # Use local var instead of db field
        sources=sources,
        code_executions=code_executions,
        created_at=new_message.created_at.isoformat(),
    )


async def set_message_feedback(
    data: MessageFeedback, user_id: int, db: AsyncSession
) -> None:
    result = await db.execute(
        select(Message)
        .join(Dialogue, Message.dialogue_id == Dialogue.id)
        .where(Message.id == data.message_id, Dialogue.user_id == user_id)
    )
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Message not found"
        )

    message.feedback = data.feedback.value

    await db.commit()


async def _call_external_rag_api(message: str, user_id: int) -> tuple[str, list[str]]:
    """Fallback to external RAG API.

    Args:
        message: User message
        user_id: User ID

    Returns:
        Tuple of (assistant_response, sources)

    Raises:
        HTTPException: If external API fails
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            rag_response = await client.post(
                RAG_API_URL, json={"text": message, "tg_user_id": user_id}
            )
            rag_response.raise_for_status()
            rag_data = rag_response.json()

            assistant_response = rag_data.get("response", "")
            sources = extract_sources_from_response(assistant_response)

            return assistant_response, sources

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG API unavailable: {str(e)}",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG API error: {e.response.status_code}",
        )


def extract_sources_from_response(response: str) -> list[str]:
    """Extract source citations from response text.

    Looks for [§N] citations and converts them to PyTorch doc URLs.

    Args:
        response: Assistant response text

    Returns:
        List of source URLs
    """
    import re

    sources = []
    citation_pattern = r"\[§(\d+)\]"
    citations = re.findall(citation_pattern, response)

    if citations:
        sources = [
            f"https://pytorch.org/docs/stable/source_{n}.html" for n in set(citations)
        ]

    return sources
