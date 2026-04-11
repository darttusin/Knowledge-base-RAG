from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from db import Dialogue, Message, Source, Folder
from settings import settings
from services.rag_service import get_rag_service

from .code_parser import parse_and_execute_code
from .models import CodeExecution, MessageFeedback, MessageResponse, SendMessage, SourceReference


async def find_source_by_path(path: str, user_id: int, db: AsyncSession) -> Source | None:
    """Find Source in database by path from ChromaDB.

    Args:
        path: Path from ChromaDB (e.g., "folder/subfolder/file.md")
        user_id: User ID to filter sources
        db: Database session

    Returns:
        Source object or None if not found
    """
    # Parse path into folder_path and filename
    if "/" in path:
        folder_path_str = "/" + path.rsplit("/", 1)[0]
        filename = path.rsplit("/", 1)[1]
    else:
        folder_path_str = None
        filename = path

    # Find folder by path if needed
    folder_id = None
    if folder_path_str:
        folder_result = await db.execute(
            select(Folder).where(
                Folder.user_id == user_id,
                Folder.path == folder_path_str
            )
        )
        folder = folder_result.scalar_one_or_none()
        if folder:
            folder_id = folder.id

    # Find source by name and folder_id
    result = await db.execute(
        select(Source)
        .options(joinedload(Source.folder))
        .where(
            Source.user_id == user_id,
            Source.name == filename,
            Source.folder_id == folder_id
        )
    )
    return result.scalar_one_or_none()


def calculate_smart_relevance(
    chunks_by_source: dict[int, list[tuple[int, float, str]]]
) -> dict[int, tuple[float, str]]:
    """Calculate smart relevance score for each source.

    Args:
        chunks_by_source: Dict of source_id -> list of (position, score, chunk_text)

    Returns:
        Dict of source_id -> (relevance_score, best_chunk_text)
    """
    import math

    total_chunks = sum(len(chunks) for chunks in chunks_by_source.values())
    results = {}

    for source_id, chunks in chunks_by_source.items():
        # Sort chunks by position
        chunks_sorted = sorted(chunks, key=lambda x: x[0])

        # 1. Weighted average score (40%)
        weighted_sum = 0.0
        weight_sum = 0.0
        for position, score, _ in chunks_sorted:
            weight = 1.0 / position
            weighted_sum += score * weight
            weight_sum += weight
        weighted_avg_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        # 2. Position score (30%) - exponential decay
        best_position = chunks_sorted[0][0]
        position_score = math.exp(-0.5 * (best_position - 1))

        # 3. Frequency score (30%) - saturates after 3+ chunks
        frequency_score = min(len(chunks) / 3.0, 1.0)

        # Combined score
        relevance_score = (
            weighted_avg_score * 0.4 +
            position_score * 0.3 +
            frequency_score * 0.3
        )

        # Use chunk with best individual score for display
        best_chunk = max(chunks_sorted, key=lambda x: x[1])

        results[source_id] = (relevance_score, best_chunk[2])

    return results


async def send_message(
    data: SendMessage, user_id: int, db: AsyncSession
) -> MessageResponse:
    # Load dialogue with messages to check if it's first message
    result = await db.execute(
        select(Dialogue)
        .where(Dialogue.id == data.dialogue_id, Dialogue.user_id == user_id)
        .options(selectinload(Dialogue.messages))
    )
    dialogue = result.scalar_one_or_none()

    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )

    is_first_message = len(dialogue.messages) == 0

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

        # Group chunks by source and collect metadata
        chunks_by_source: dict[int, list[tuple[int, float, str]]] = {}  # source_id -> [(position, score, text)]
        source_metadata: dict[int, tuple[str, str | None]] = {}  # source_id -> (name, folder_path)

        for position, chunk in enumerate(rag_response.chunks, start=1):
            # Clean source path by removing common prefixes
            source_path = chunk.source
            for prefix in ["drive/MyDrive/dataset/", "dataset/"]:
                if source_path.startswith(prefix):
                    source_path = source_path[len(prefix):]
                    break

            # Find source in database
            source_obj = await find_source_by_path(source_path, user_id, db)

            if source_obj:
                # Store chunk data
                if source_obj.id not in chunks_by_source:
                    chunks_by_source[source_obj.id] = []
                    folder_path = source_obj.folder.path if source_obj.folder else None
                    source_metadata[source_obj.id] = (source_obj.name, folder_path)

                chunks_by_source[source_obj.id].append((position, chunk.score, chunk.text))

        # Calculate smart relevance scores
        relevance_scores = calculate_smart_relevance(chunks_by_source)

        # Build source references
        source_references = []
        for source_id, (relevance_score, best_chunk_text) in relevance_scores.items():
            document_name, folder_path = source_metadata[source_id]
            source_references.append(
                SourceReference(
                    source_id=source_id,
                    document_name=document_name,
                    chunk_text=best_chunk_text,
                    relevance_score=relevance_score,
                    folder_path=folder_path
                )
            )

        # Sort by relevance score
        source_references = sorted(
            source_references,
            key=lambda x: x.relevance_score,
            reverse=True
        )

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

    # Log clean answer without markdown/formatting
    from loguru import logger
    logger.info(f"RAG Response | User: {data.message[:100]}... | Answer: {assistant_response}")

    # Save source references as JSON array for database record
    import json
    sources_json = [src.model_dump() for src in source_references]
    new_message.sources = json.dumps(sources_json) if sources_json else None

    # Generate dialogue title if this is first message and title is still default/empty
    normalized_dialogue_name = (dialogue.name or "").strip().lower()
    should_generate_title = (
        is_first_message
        and normalized_dialogue_name in {"", "new conversation"}
    )

    if should_generate_title:
        from api.dialogue.controller import generate_dialogue_title
        try:
            new_title = generate_dialogue_title(data.message, rag_service)
            dialogue.name = new_title
        except Exception:
            # Keep default name on error
            pass

    await db.commit()
    await db.refresh(new_message)

    code_results = await parse_and_execute_code(assistant_response, settings.CODE_EXECUTOR_URL)
    code_executions = [CodeExecution(**result) for result in code_results]

    return MessageResponse(
        message_id=new_message.id,
        user_message=new_message.user_message,
        assistant_response=assistant_response,
        sources=source_references,
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

    # Log feedback for future analysis and fine-tuning
    from loguru import logger
    logger.info(
        f"User feedback: {data.feedback.value} | "
        f"Message ID: {data.message_id} | "
        f"User ID: {user_id} | "
        f"Question: {message.user_message[:100]}... | "
        f"Answer: {message.assistant_response[:100] if message.assistant_response else 'N/A'}..."
    )

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
