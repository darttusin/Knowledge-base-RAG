"""
Роутер для эмбеддинга документов
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from services.embedding_service import embed_document, delete_document_embeddings, delete_user_embeddings

router = APIRouter(prefix="/embed", tags=["Embedding"])


class EmbedDocumentRequest(BaseModel):
    """Запрос на эмбеддинг документа"""
    document_id: int
    document_name: str
    content: str
    user_id: int
    chunk_size: int = 500
    overlap: int = 50


class EmbedDocumentResponse(BaseModel):
    """Ответ на эмбеддинг документа"""
    success: bool
    chunks_count: int
    document_id: int
    document_name: str
    error: str | None = None


class DeleteEmbeddingsRequest(BaseModel):
    """Запрос на удаление эмбеддингов"""
    document_id: int


class DeleteEmbeddingsResponse(BaseModel):
    """Ответ на удаление эмбеддингов"""
    success: bool
    deleted_count: int
    error: str | None = None


@router.post(
    "/document",
    response_model=EmbedDocumentResponse,
    summary="Эмбеддить документ",
    description="Разбивает документ на чанки, создает эмбеддинги и сохраняет в ChromaDB"
)
async def embed_document_endpoint(request: EmbedDocumentRequest) -> EmbedDocumentResponse:
    """
    Эмбеддит документ и сохраняет в ChromaDB
    """
    result = embed_document(
        document_id=request.document_id,
        document_name=request.document_name,
        content=request.content,
        user_id=request.user_id,
        chunk_size=request.chunk_size,
        overlap=request.overlap
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to embed document")
        )

    return EmbedDocumentResponse(**result)


@router.delete(
    "/document/{document_id}",
    response_model=DeleteEmbeddingsResponse,
    summary="Удалить эмбеддинги документа",
    description="Удаляет все эмбеддинги документа из ChromaDB"
)
async def delete_embeddings_endpoint(document_id: int) -> DeleteEmbeddingsResponse:
    """
    Удаляет все эмбеддинги документа
    """
    result = delete_document_embeddings(document_id)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to delete embeddings")
        )

    return DeleteEmbeddingsResponse(**result)


@router.delete(
    "/user/{user_id}",
    response_model=DeleteEmbeddingsResponse,
    summary="Удалить все эмбеддинги пользователя",
    description="Удаляет все эмбеддинги пользователя из ChromaDB"
)
async def delete_user_embeddings_endpoint(user_id: int) -> DeleteEmbeddingsResponse:
    """
    Удаляет все эмбеддинги пользователя
    """
    result = delete_user_embeddings(user_id)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("error", "Failed to delete embeddings")
        )

    return DeleteEmbeddingsResponse(**result)
