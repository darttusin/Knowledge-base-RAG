from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.folder.schema import FolderCreateRequest, FolderResponse
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
