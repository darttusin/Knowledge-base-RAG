from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.folder.schema import FolderCreateRequest, FolderMoveRequest, FolderResponse
from auth import get_current_user_id
from db import Folder, Source, get_db

router = APIRouter(prefix="/api/folder", tags=["folders"])


@router.post("", response_model=FolderResponse, status_code=status.HTTP_201_CREATED)
async def create_folder(
    request: FolderCreateRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Create a new folder"""

    # Check if folder with same path already exists for this user
    result = await db.execute(
        select(Folder).where(
            Folder.user_id == user_id,
            Folder.path == request.path
        )
    )
    existing_folder = result.scalar_one_or_none()

    if existing_folder:
        return FolderResponse(
            id=existing_folder.id,
            name=existing_folder.name,
            path=existing_folder.path,
            parent_id=existing_folder.parent_id,
            created_at=existing_folder.created_at,
        )

    new_folder = Folder(
        user_id=user_id,
        name=request.name,
        path=request.path,
        parent_id=request.parent_id,
        created_at=datetime.utcnow(),
    )

    db.add(new_folder)
    await db.commit()
    await db.refresh(new_folder)

    return FolderResponse(
        id=new_folder.id,
        name=new_folder.name,
        path=new_folder.path,
        parent_id=new_folder.parent_id,
        created_at=new_folder.created_at,
    )


@router.get("", response_model=List[FolderResponse])
async def get_folders(
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get all folders for the current user with document counts"""
    # Get folders with document counts
    result = await db.execute(
        select(
            Folder,
            func.count(Source.id).label("doc_count")
        )
        .outerjoin(Source, Folder.id == Source.folder_id)
        .where(Folder.user_id == user_id)
        .group_by(Folder.id)
        .order_by(Folder.path)
    )

    folders_with_counts = result.all()

    return [
        FolderResponse(
            id=folder.id,
            name=folder.name,
            path=folder.path,
            parent_id=folder.parent_id,
            created_at=folder.created_at,
            document_count=doc_count,
        )
        for folder, doc_count in folders_with_counts
    ]


@router.patch("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def move_folder(
    folder_id: int,
    request: FolderMoveRequest,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Move folder under a different parent (or to root if parent_id=null)."""
    result = await db.execute(select(Folder).where(Folder.user_id == user_id))
    all_folders = list(result.scalars().all())
    folders_by_id = {f.id: f for f in all_folders}

    folder = folders_by_id.get(folder_id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
        )

    new_parent_id = request.parent_id

    if new_parent_id is not None:
        if new_parent_id == folder_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot move folder into itself",
            )
        if new_parent_id not in folders_by_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Target folder not found"
            )
        # Walk up from new parent to ensure the folder isn't one of its ancestors
        cursor = folders_by_id[new_parent_id]
        while cursor.parent_id is not None:
            if cursor.parent_id == folder_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot move folder into its own descendant",
                )
            cursor = folders_by_id.get(cursor.parent_id)
            if cursor is None:
                break

    if folder.parent_id == new_parent_id:
        return

    folder.parent_id = new_parent_id

    # Rebuild paths for the moved folder and all of its descendants (BFS).
    children_map: dict[int | None, list[Folder]] = {}
    for f in all_folders:
        children_map.setdefault(f.parent_id, []).append(f)

    queue: list[Folder] = [folder]
    while queue:
        current = queue.pop(0)
        parent_path = (
            folders_by_id[current.parent_id].path
            if current.parent_id is not None
            else ""
        )
        current.path = f"{parent_path}/{current.name}"
        for child in children_map.get(current.id, []):
            queue.append(child)

    await db.commit()


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: int,
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Delete a folder (and all its contents)"""
    result = await db.execute(
        select(Folder).where(Folder.id == folder_id, Folder.user_id == user_id)
    )
    folder = result.scalar_one_or_none()

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )

    await db.delete(folder)
    await db.commit()
