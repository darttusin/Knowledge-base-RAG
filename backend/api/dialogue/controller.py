from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import Dialogue

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
            query="What's the difference between torch.nn.Module and torch.nn.functional?",
            icon=IconsEnum.database,
        ),
    ]


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
                parent_message_id=msg.parent_message_id,
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
