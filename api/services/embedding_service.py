"""
Сервис для эмбеддинга документов и сохранения в ChromaDB
"""
import re
from typing import List
from uuid import uuid4

from services.rag_service import COLLECTION, MODEL


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием

    Args:
        text: Исходный текст
        chunk_size: Размер чанка в символах
        overlap: Перекрытие между чанками

    Returns:
        Список чанков
    """
    # Разбиваем по параграфам
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Если параграф пустой, пропускаем
        if not paragraph.strip():
            continue

        # Если текущий чанк + параграф не превышает размер
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            # Сохраняем текущий чанк
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Начинаем новый чанк с перекрытием
            if overlap > 0 and current_chunk:
                # Берем последние overlap символов предыдущего чанка
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + paragraph + "\n\n"
            else:
                current_chunk = paragraph + "\n\n"

    # Добавляем последний чанк
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def embed_document(
    document_id: int,
    document_name: str,
    content: str,
    user_id: int,
    chunk_size: int = 500,
    overlap: int = 50
) -> dict:
    """
    Эмбеддит документ и сохраняет в ChromaDB

    Args:
        document_id: ID документа в БД
        document_name: Имя документа
        content: Содержимое документа
        user_id: ID пользователя
        chunk_size: Размер чанка
        overlap: Перекрытие между чанками

    Returns:
        Словарь с информацией о результате
    """
    # Разбиваем на чанки
    chunks = chunk_text(content, chunk_size, overlap)

    if not chunks:
        return {
            "success": False,
            "error": "No chunks created from document",
            "chunks_count": 0
        }

    # Генерируем embeddings
    embeddings = MODEL.encode(chunks).tolist()

    # Создаем ID для каждого чанка
    ids = [f"doc_{document_id}_chunk_{i}_{uuid4().hex[:8]}" for i in range(len(chunks))]

    # Метаданные для каждого чанка
    metadatas = [
        {
            "source": document_name,
            "document_id": document_id,
            "chunk_index": i,
            "user_id": user_id,
            "type": "document"
        }
        for i in range(len(chunks))
    ]

    # Сохраняем в ChromaDB
    try:
        COLLECTION.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return {
            "success": True,
            "chunks_count": len(chunks),
            "document_id": document_id,
            "document_name": document_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "chunks_count": 0
        }


def delete_document_embeddings(document_id: int) -> dict:
    """
    Удаляет все эмбеддинги документа из ChromaDB

    Args:
        document_id: ID документа

    Returns:
        Словарь с информацией о результате
    """
    try:
        # Ищем все чанки документа
        results = COLLECTION.get(
            where={"document_id": document_id}
        )

        if results["ids"]:
            # Удаляем все найденные чанки
            COLLECTION.delete(ids=results["ids"])

            return {
                "success": True,
                "deleted_count": len(results["ids"])
            }
        else:
            return {
                "success": True,
                "deleted_count": 0
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "deleted_count": 0
        }


def delete_user_embeddings(user_id: int) -> dict:
    """
    Удаляет все эмбеддинги пользователя из ChromaDB

    Args:
        user_id: ID пользователя

    Returns:
        Словарь с информацией о результате
    """
    try:
        # Ищем все чанки пользователя
        results = COLLECTION.get(
            where={"user_id": user_id}
        )

        if results["ids"]:
            # Удаляем все найденные чанки
            COLLECTION.delete(ids=results["ids"])

            return {
                "success": True,
                "deleted_count": len(results["ids"])
            }
        else:
            return {
                "success": True,
                "deleted_count": 0
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "deleted_count": 0
        }
