import math

from fastapi import HTTPException, UploadFile, status
from loguru import logger
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import Folder, Source
from services.rag_service import get_rag_service
from settings import settings as app_settings

from .models import SourceContent, SourceForList, SourcesList, SourceType

MAX_FILE_SIZE = 10 * 1024 * 1024


async def upload_source(
    file: UploadFile, user_id: int, db: AsyncSession
) -> SourceForList:
    content = await file.read()
    size_bytes = len(content)

    if size_bytes > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024} MB",
        )

    if size_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty"
        )

    filename = file.filename or "unknown.txt"
    file_extension = filename.split(".")[-1].lower() if "." in filename else "txt"

    try:
        source_type = SourceType(file_extension)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_extension}. Supported: {[t.value for t in SourceType]}",
        )

    try:
        content_str = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be UTF-8 encoded text",
        )

    new_source = Source(
        user_id=user_id,
        name=filename,
        source_type=source_type.value,
        content=content_str,
        size_bytes=size_bytes,
    )

    db.add(new_source)
    await db.commit()
    await db.refresh(new_source)

    # Embed document using local RAG service
    if app_settings.RAG_ENABLED:
        try:
            from rag.documents import Document, split_documents
            from rag.vectorstore import index_chunks

            rag_service = get_rag_service()

            # Create document object
            doc = Document(
                page_content=content_str,
                metadata={
                    "source": filename,
                    "document_id": new_source.id,
                    "user_id": user_id,
                    "type": source_type.value,
                }
            )

            # Split document into chunks
            chunks = split_documents(
                [doc],
                chunk_size=app_settings.RAG_CHUNK_SIZE,
                chunk_overlap=app_settings.RAG_CHUNK_OVERLAP,
            )

            # Index chunks in ChromaDB
            index_chunks(rag_service.collection, chunks, rag_service.embed_model)

            logger.info(f"✓ Embedded document {new_source.id}: {len(chunks)} chunks indexed")

        except Exception as e:
            # If embedding fails, log but don't fail the upload
            logger.warning(f"⚠ Failed to embed document {new_source.id}: {str(e)}")
    else:
        logger.warning(f"⚠ RAG disabled, skipping document embedding for {new_source.id}")

    return SourceForList(
        source_id=new_source.id,
        name=new_source.name,
        source_type=SourceType(new_source.source_type),
        size_bytes=new_source.size_bytes,
        created_at=new_source.created_at.isoformat(),
        folder_id=new_source.folder_id,
        folder_path=None,
    )


async def get_sources_list(
    user_id: int,
    query: str | None,
    folder_id: int | None,
    page: int,
    limit: int,
    search_in: str,
    db: AsyncSession
) -> SourcesList:
    stmt = select(Source).where(Source.user_id == user_id).options(selectinload(Source.folder))

    # Full-text search if query provided
    if query and query.strip():
        # Prepare tsquery (escape single quotes and special chars)
        tsquery = query.strip().replace("'", "''")

        if search_in == "name":
            # Search only in name using FTS
            stmt = stmt.where(
                text("name_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
            # Order by relevance
            stmt = stmt.order_by(
                text("ts_rank(name_tsvector, plainto_tsquery('english', :query)) DESC")
            ).params(query=tsquery)
        elif search_in == "content":
            # Search only in content using FTS
            stmt = stmt.where(
                text("content_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
            # Order by relevance
            stmt = stmt.order_by(
                text("ts_rank(content_tsvector, plainto_tsquery('english', :query)) DESC")
            ).params(query=tsquery)
        else:  # both
            # Search in both name and content
            stmt = stmt.where(
                text("""
                    name_tsvector @@ plainto_tsquery('english', :query) OR
                    content_tsvector @@ plainto_tsquery('english', :query)
                """)
            ).params(query=tsquery)
            # Order by combined relevance
            stmt = stmt.order_by(
                text("""
                    (ts_rank(name_tsvector, plainto_tsquery('english', :query)) * 2.0 +
                     ts_rank(content_tsvector, plainto_tsquery('english', :query))) DESC
                """)
            ).params(query=tsquery)
    else:
        # No search query - just sort by date
        stmt = stmt.order_by(Source.created_at.desc())

    if folder_id is not None:
        stmt = stmt.where(Source.folder_id == folder_id)

    # Count total matching documents
    count_stmt = select(func.count(Source.id)).where(Source.user_id == user_id)
    if query and query.strip():
        tsquery = query.strip().replace("'", "''")
        if search_in == "name":
            count_stmt = count_stmt.where(
                text("name_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
        elif search_in == "content":
            count_stmt = count_stmt.where(
                text("content_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
        else:
            count_stmt = count_stmt.where(
                text("""
                    name_tsvector @@ plainto_tsquery('english', :query) OR
                    content_tsvector @@ plainto_tsquery('english', :query)
                """)
            ).params(query=tsquery)
    if folder_id is not None:
        count_stmt = count_stmt.where(Source.folder_id == folder_id)
    total_count_result = await db.execute(count_stmt)
    total_count = total_count_result.scalar() or 0

    # Calculate total size
    size_stmt = select(func.sum(Source.size_bytes)).where(Source.user_id == user_id)
    if query and query.strip():
        tsquery = query.strip().replace("'", "''")
        if search_in == "name":
            size_stmt = size_stmt.where(
                text("name_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
        elif search_in == "content":
            size_stmt = size_stmt.where(
                text("content_tsvector @@ plainto_tsquery('english', :query)")
            ).params(query=tsquery)
        else:
            size_stmt = size_stmt.where(
                text("""
                    name_tsvector @@ plainto_tsquery('english', :query) OR
                    content_tsvector @@ plainto_tsquery('english', :query)
                """)
            ).params(query=tsquery)
    if folder_id is not None:
        size_stmt = size_stmt.where(Source.folder_id == folder_id)
    total_size_result = await db.execute(size_stmt)
    total_size_bytes = total_size_result.scalar() or 0

    # Apply pagination
    offset = (page - 1) * limit
    stmt = stmt.offset(offset).limit(limit)

    result = await db.execute(stmt)
    sources = result.scalars().all()

    total_pages = math.ceil(total_count / limit) if limit > 0 else 0

    return SourcesList(
        total_count=total_count,
        total_size_bytes=total_size_bytes,
        page=page,
        limit=limit,
        total_pages=total_pages,
        sources=[
            SourceForList(
                source_id=source.id,
                name=source.name,
                source_type=SourceType(source.source_type),
                size_bytes=source.size_bytes,
                created_at=source.created_at.isoformat(),
                folder_id=source.folder_id,
                folder_path=source.folder.path if source.folder else None,
            )
            for source in sources
        ],
    )


async def get_source(source_id: int, user_id: int, db: AsyncSession) -> SourceContent:
    result = await db.execute(
        select(Source)
        .where(Source.id == source_id, Source.user_id == user_id)
        .options(selectinload(Source.folder))
    )
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source not found"
        )

    return SourceContent(
        source_id=source.id,
        name=source.name,
        source_type=SourceType(source.source_type),
        content=source.content,
        size_bytes=source.size_bytes,
        created_at=source.created_at.isoformat(),
        folder_id=source.folder_id,
        folder_path=source.folder.path if source.folder else None,
    )


async def move_source(
    source_id: int, folder_id: int | None, user_id: int, db: AsyncSession
) -> None:
    result = await db.execute(
        select(Source).where(Source.id == source_id, Source.user_id == user_id)
    )
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source not found"
        )

    if folder_id is not None:
        result = await db.execute(
            select(Folder).where(Folder.id == folder_id, Folder.user_id == user_id)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
            )

    source.folder_id = folder_id
    await db.commit()


async def delete_source(source_id: int, user_id: int, db: AsyncSession) -> None:
    result = await db.execute(
        select(Source).where(Source.id == source_id, Source.user_id == user_id)
    )
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source not found"
        )

    # Delete embeddings from ChromaDB
    if app_settings.RAG_ENABLED:
        try:
            rag_service = get_rag_service()

            # Delete all chunks for this document from ChromaDB
            # We filter by document_id in metadata
            rag_service.collection.delete(
                where={"document_id": source_id}
            )

            logger.info(f"✓ Deleted embeddings for document {source_id}")

        except Exception as e:
            # Log but continue with document deletion
            logger.warning(f"⚠ Failed to delete embeddings for document {source_id}: {str(e)}")
    else:
        logger.warning(f"⚠ RAG disabled, skipping embedding deletion for {source_id}")

    await db.delete(source)
    await db.commit()


async def download_source(
    source_id: int, user_id: int, db: AsyncSession
) -> tuple[str, str, str]:
    result = await db.execute(
        select(Source).where(Source.id == source_id, Source.user_id == user_id)
    )
    source = result.scalar_one_or_none()

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Source not found"
        )

    media_type_map = {
        "txt": "text/plain",
        "md": "text/markdown",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_type_map.get(source.source_type, "text/plain")

    return source.name, source.content, media_type
