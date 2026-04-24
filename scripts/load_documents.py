#!/usr/bin/env python3
"""
Скрипт для загрузки всех документов из data/dataset в базу как источники
"""
import sys
import asyncio
import httpx
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Загружаем .env файл из backend
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"
load_dotenv(env_path)

# Добавляем корневую директорию в PATH
sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from db import init_db, close_db, get_db, Source, User

DATASET_PATH = Path(__file__).parent.parent / "data" / "dataset"
ALLOWED_EXTENSIONS = {".md", ".txt"}
RAG_API_URL = "http://localhost:8000"


async def load_documents(user_id: int):
    """Загрузить все документы из папки dataset"""
    await init_db()

    if not DATASET_PATH.exists():
        print(f"❌ Папка {DATASET_PATH} не найдена")
        return

    print(f"📂 Сканирование папки: {DATASET_PATH}")

    async for db in get_db():
        loaded_count = 0
        skipped_count = 0

        # Рекурсивно обходим все файлы
        for file_path in DATASET_PATH.rglob("*"):
            if not file_path.is_file():
                continue

            # Проверяем расширение
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            # Проверяем, существует ли уже этот файл
            result = await db.execute(
                select(Source).where(
                    Source.name == file_path.name,
                    Source.user_id == user_id
                )
            )
            existing_source = result.scalar_one_or_none()

            if existing_source:
                print(f"⏭️  Пропускаем {file_path.name} (уже существует)")
                skipped_count += 1
                continue

            # Читаем содержимое файла
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"❌ Ошибка чтения {file_path.name}: {e}")
                continue

            # Определяем тип файла
            source_type = file_path.suffix.lower().replace(".", "")

            # Создаем источник
            new_source = Source(
                user_id=user_id,
                name=file_path.name,
                source_type=source_type,
                content=content,
                size_bytes=len(content.encode("utf-8")),
                created_at=datetime.utcnow()
            )

            db.add(new_source)
            await db.flush()  # Получаем ID сразу

            # Вызываем API для эмбеддинга
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    embed_response = await client.post(
                        f"{RAG_API_URL}/embed/document",
                        json={
                            "document_id": new_source.id,
                            "document_name": new_source.name,
                            "content": content,
                            "user_id": user_id,
                            "chunk_size": 500,
                            "overlap": 50
                        }
                    )
                    embed_response.raise_for_status()
                    embed_data = embed_response.json()
                    chunks_count = embed_data.get("chunks_count", 0)
                    print(f"✅ Загружен: {file_path.name} ({len(content)} символов, {chunks_count} чанков)")
            except Exception as e:
                print(f"⚠️  Загружен: {file_path.name}, но эмбеддинг не удался: {str(e)}")

            loaded_count += 1

        # Сохраняем все изменения
        await db.commit()

        print(f"\n📊 Статистика:")
        print(f"   ✅ Загружено: {loaded_count}")
        print(f"   ⏭️  Пропущено: {skipped_count}")
        print(f"   📚 Всего: {loaded_count + skipped_count}")

    await close_db()


async def main():
    """Главная функция"""
    print("🚀 Загрузка документов из data/dataset...")

    await init_db()

    # Находим первого пользователя
    async for db in get_db():
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()

        if not user:
            print("❌ Пользователь не найден. Сначала создайте пользователя:")
            print("   python scripts/create_test_user.py")
            return

        print(f"👤 Загружаем документы для пользователя: {user.email}")
        await load_documents(user.id)

    await close_db()
    print("\n✨ Готово!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. API запущен на порту 8000")
        print("  3. Файл backend/.env существует")
        sys.exit(1)
