from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import Dialogue, Message
from settings import settings
from services.rag_service import get_rag_service

from .code_parser import parse_and_execute_code
from .models import CodeExecution, MessageFeedback, MessageResponse, SendMessage


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

    # Use local RAG service
    if not settings.RAG_ENABLED:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is disabled. Enable RAG_ENABLED in settings.",
        )

    try:
        rag_service = get_rag_service()

        # Answer question using local RAG
        rag_response = rag_service.answer_question(
            question=data.message,
            strategy="query_transform",  # Full RAG with query rewriting + HyDE
            check_topic=settings.OUTLIER_DETECTION_ENABLED,
            reject_off_topic=settings.OUTLIER_REJECT_OFF_TOPIC,
        )

        assistant_response = rag_response.answer
        sources = [chunk.source for chunk in rag_response.chunks]

    except RuntimeError as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"RAG service not initialized: {str(e)}",
        )

    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG service error: {str(e)}",
        )

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
