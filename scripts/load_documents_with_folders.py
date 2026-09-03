#!/usr/bin/env python3
"""Legacy folder importer with mixed API/direct-DB writes.

It selects the first user, assumes the fixed password ``password``, creates folders
through backend HTTP, inserts sources directly in PostgreSQL and calls the absent
``localhost:8000/embed/document`` endpoint. Failed embedding still leaves rows to
be committed; partial folder creation is not rolled back, and a root-file branch
can reference an unbound folder-path variable. Do not use as a current loader.
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
from db import init_db, close_db, get_db, Source, Folder, User

DATASET_PATH = Path(__file__).parent.parent / "data" / "dataset"
ALLOWED_EXTENSIONS = {".md", ".txt"}
BACKEND_API_URL = "http://localhost:8001"
RAG_API_URL = "http://localhost:8000"


async def get_auth_token(email: str, password: str) -> str:
    """Получить JWT токен для аутентификации"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BACKEND_API_URL}/api/user/auth",
            json={"email": email, "password": password}
        )
        response.raise_for_status()
        data = response.json()
        return data["access_token"]


async def create_folder(token: str, name: str, path: str, parent_id: int | None = None) -> int:
    """Создать папку через API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BACKEND_API_URL}/api/folder",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": name,
                "path": path,
                "parent_id": parent_id
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["id"]


async def load_documents_with_folders(user_id: int, token: str):
    """Загрузить все документы с воспроизведением структуры папок"""
    await init_db()

    if not DATASET_PATH.exists():
        print(f"❌ Папка {DATASET_PATH} не найдена")
        return

    print(f"📂 Сканирование папки: {DATASET_PATH}")

    # Собираем все папки и создаем их в базе
    folders_map = {}  # path -> folder_id

    print("\n📁 Создание структуры папок...")
    all_folders = sorted([d for d in DATASET_PATH.rglob("*") if d.is_dir()])

    for folder_path in all_folders:
        # Получаем относительный путь от dataset
        rel_path = folder_path.relative_to(DATASET_PATH)
        folder_name = folder_path.name
        folder_path_str = "/" + str(rel_path).replace("\\", "/")

        # Определяем parent_id
        parent_id = None
        if rel_path.parent != Path("."):
            parent_path = "/" + str(rel_path.parent).replace("\\", "/")
            parent_id = folders_map.get(parent_path)

        # Создаем папку
        try:
            folder_id = await create_folder(token, folder_name, folder_path_str, parent_id)
            folders_map[folder_path_str] = folder_id
            print(f"   ✅ Папка создана: {folder_path_str}")
        except Exception as e:
            print(f"   ⚠️  Ошибка создания папки {folder_path_str}: {str(e)}")

    # Теперь загружаем документы
    print(f"\n📄 Загрузка документов...")

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

            # Определяем folder_id
            folder_id = None
            rel_path = file_path.relative_to(DATASET_PATH)
            if rel_path.parent != Path("."):
                folder_path_str = "/" + str(rel_path.parent).replace("\\", "/")
                folder_id = folders_map.get(folder_path_str)

            # Проверяем, существует ли уже этот файл в этой папке
            result = await db.execute(
                select(Source).where(
                    Source.name == file_path.name,
                    Source.folder_id == folder_id,
                    Source.user_id == user_id
                )
            )
            existing_source = result.scalar_one_or_none()

            if existing_source:
                print(f"⏭️  Пропускаем {file_path.name} в {folder_path_str} (уже существует)")
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
                folder_id=folder_id,
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

                    folder_display = f" в папке {folder_path_str}" if folder_id else ""
                    print(f"✅ Загружен: {file_path.name}{folder_display} ({len(content)} символов, {chunks_count} чанков)")
            except Exception as e:
                print(f"⚠️  Загружен: {file_path.name}, но эмбеддинг не удался: {str(e)}")

            loaded_count += 1

        # Сохраняем все изменения
        await db.commit()

        print(f"\n📊 Статистика:")
        print(f"   📁 Папок создано: {len(folders_map)}")
        print(f"   ✅ Документов загружено: {loaded_count}")
        print(f"   ⏭️  Документов пропущено: {skipped_count}")
        print(f"   📚 Всего: {loaded_count + skipped_count}")

    await close_db()


async def main():
    """Главная функция"""
    print("🚀 Загрузка документов из data/dataset с воспроизведением структуры папок...")

    await init_db()

    # Находим первого пользователя
    async for db in get_db():
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()

        if not user:
            print("❌ Пользователь не найден. Сначала создайте пользователя:")
            print("   uv run scripts/create_test_user.py")
            return

        print(f"👤 Загружаем документы для пользователя: {user.email}")

        # Получаем токен
        try:
            token = await get_auth_token(user.email, "password")
        except Exception as e:
            print(f"❌ Не удалось получить токен аутентификации: {e}")
            print("   Проверьте, что backend запущен на порту 8001")
            return

        await load_documents_with_folders(user.id, token)

    await close_db()
    print("\n✨ Готово!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. Backend запущен на порту 8001")
        print("  3. API запущен на порту 8000")
        print("  4. Файл backend/.env существует")
        sys.exit(1)
