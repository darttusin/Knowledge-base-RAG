import re
from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import Dialogue
from services.rag_service import RagService

from .models import (
    CreateDialogue,
    DialogueResponse,
    IconsEnum,
    MessageResponse,
    PreGeneratedQuery,
    ShortDialogue,
    SourceReference,
    UpdateDialogue,
)


def generate_pre_generated_queries() -> list[PreGeneratedQuery]:
    return [
        PreGeneratedQuery(
            query="How does torch.autograd work for automatic differentiation?",
            icon=IconsEnum.doc,
        ),
        PreGeneratedQuery(
            query="What's the difference between torch.nn.Module and torch.nn.functional?",
            icon=IconsEnum.database,
        ),
        PreGeneratedQuery(
            query="How to properly configure a torch learning rate scheduler?",
            icon=IconsEnum.browser,
        ),
    ]


def generate_dialogue_title(question: str, rag_service: RagService) -> str:
    """Generate dialogue title from first question using LLM.

    Args:
        question: User's first question
        rag_service: RAG service instance with chat model

    Returns:
        Generated title (3-6 words)
    """
    prompt = f"""Generate a short, concise title (3-6 words) for a conversation that starts with this question.
The title should capture the main topic or intent of the question.
Output ONLY the title, nothing else.

Question: {question}
Title:"""

    def _fallback_title() -> str:
        words = question.split()[:6]
        return " ".join(words) if words else "New conversation"

    def _extract_stream_content(chunk: object) -> tuple[str, bool]:
        if isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if not choices:
                return "", False
            choice = choices[0] or {}
            delta = choice.get("delta", {}) or {}
            content = delta.get("content", "") or ""
            return content, bool(choice.get("finish_reason"))

        choices = getattr(chunk, "choices", None)
        if not choices:
            return "", False
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        content = getattr(delta, "content", "") if delta else ""
        finish_reason = getattr(choice, "finish_reason", None)
        return content or "", bool(finish_reason)

    def _normalize_title(raw_title: str) -> str:
        title = raw_title.strip()
        title = re.sub(r"[\r\n]+", " ", title)
        title = title.strip().strip("\"'`")
        title = re.sub(
            r"^(title|topic|summary|заголовок)\s*[:\-]\s*",
            "",
            title,
            flags=re.IGNORECASE,
        )
        title = title.replace('"', "").replace("'", "").replace("`", "")
        title = re.sub(r"\s+", " ", title).strip()

        if not title:
            return ""

        words = title.split()
        if len(words) > 6:
            words = words[:6]
        if len(words) < 3:
            return ""
        return " ".join(words)

    try:
        messages = [{"role": "user", "content": prompt}]
        chat_model = rag_service.chat_model

        if hasattr(chat_model, "invoke_once"):
            response = chat_model.invoke_once(
                messages, max_tokens=16, temperature=0.2
            )
        else:
            response = chat_model.invoke(messages, max_tokens=16, temperature=0.2)

        if isinstance(response, str):
            title = _normalize_title(response)
            return title if title else _fallback_title()

        if isinstance(response, Iterable):
            chunks: list[str] = []
            for chunk in response:
                content, is_finished = _extract_stream_content(chunk)
                if content:
                    chunks.append(content)
                if is_finished:
                    break
            title = _normalize_title("".join(chunks))
            return title if title else _fallback_title()

        return _fallback_title()
    except Exception:
        return _fallback_title()


async def create_dialogue(
    data: CreateDialogue, user_id: int, db: AsyncSession
) -> DialogueResponse:
    new_dialogue = Dialogue(user_id=user_id, name=data.name)

    db.add(new_dialogue)
    await db.commit()
    await db.refresh(new_dialogue)

    pre_generated = generate_pre_generated_queries()

    return DialogueResponse(
        dialogue_id=new_dialogue.id,
        name=new_dialogue.name,
        created_at=new_dialogue.created_at.isoformat(),
        updated_at=new_dialogue.updated_at.isoformat(),
        pre_generated_queries=pre_generated,
        messages=[],
    )


async def get_dialogue(
    dialogue_id: int, user_id: int, db: AsyncSession
) -> DialogueResponse:
    result = await db.execute(
        select(Dialogue)
        .where(Dialogue.id == dialogue_id, Dialogue.user_id == user_id)
        .options(selectinload(Dialogue.messages))
    )
    dialogue = result.scalar_one_or_none()

    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )

    pre_generated = generate_pre_generated_queries()

    import json

    messages = []
    for msg in dialogue.messages:
        # Parse sources from JSON
        sources = None
        if msg.sources:
            try:
                sources_data = json.loads(msg.sources)
                # Handle both old format (list of strings) and new format (list of SourceReference dicts)
                if sources_data and isinstance(sources_data[0], dict):
                    sources = [SourceReference(**src) for src in sources_data]
                else:
                    # Old format - list of paths, skip for now or convert to empty
                    sources = None
            except (json.JSONDecodeError, KeyError, TypeError):
                sources = None

        messages.append(
            MessageResponse(
                message_id=msg.id,
                user_message=msg.user_message,
                assistant_response=msg.assistant_response,
                sources=sources,
                feedback=msg.feedback,
                created_at=msg.created_at.isoformat(),
            )
        )

    return DialogueResponse(
        dialogue_id=dialogue.id,
        name=dialogue.name,
        created_at=dialogue.created_at.isoformat(),
        updated_at=dialogue.updated_at.isoformat(),
        pre_generated_queries=pre_generated,
        messages=messages,
    )


async def get_dialogues(
    user_id: int, query: str | None, db: AsyncSession
) -> list[ShortDialogue]:
    stmt = select(Dialogue).where(Dialogue.user_id == user_id)

    if query:
        stmt = stmt.where(Dialogue.name.ilike(f"%{query}%"))

    stmt = stmt.order_by(Dialogue.updated_at.desc())

    result = await db.execute(stmt)
    dialogues = result.scalars().all()

    return [
        ShortDialogue(
            dialogue_id=dialogue.id,
            name=dialogue.name,
            created_at=dialogue.created_at.isoformat(),
            updated_at=dialogue.updated_at.isoformat(),
        )
        for dialogue in dialogues
    ]


async def update_dialogue(
    dialogue_id: int, changes: UpdateDialogue, user_id: int, db: AsyncSession
) -> None:
    result = await db.execute(
        select(Dialogue).where(Dialogue.id == dialogue_id, Dialogue.user_id == user_id)
    )
    dialogue = result.scalar_one_or_none()

    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )

    if changes.name is not None:
        dialogue.name = changes.name

    await db.commit()


async def delete_dialogue(dialogue_id: int, user_id: int, db: AsyncSession) -> None:
    result = await db.execute(
        select(Dialogue).where(Dialogue.id == dialogue_id, Dialogue.user_id == user_id)
    )
    dialogue = result.scalar_one_or_none()

    if not dialogue:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dialogue not found"
        )

    await db.delete(dialogue)
    await db.commit()
