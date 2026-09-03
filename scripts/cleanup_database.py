#!/usr/bin/env python3
"""Destructive legacy cleanup for the first user returned by PostgreSQL.

После интерактивного подтверждения удаляет ``Source`` и ``Folder`` только этого
пользователя. Диалоги не удаляются, а ChromaDB вообще не меняется, поэтому vectors
остаются stale. Это не согласованный reset и не подходит для shared/prod БД.
"""
import sys
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл из backend
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"
load_dotenv(env_path)

# Добавляем корневую директорию в PATH
sys.path.insert(0, str(backend_dir))

from sqlalchemy import delete
from db import init_db, close_db, get_db, Source, Folder, User

RAG_API_URL = "http://localhost:8000"


async def cleanup_database():
    """Очистить базу данных"""
    await init_db()

    async for db in get_db():
        # Получаем первого пользователя
        from sqlalchemy import select
        result = await db.execute(select(User).limit(1))
        user = result.scalar_one_or_none()

        if not user:
            print("❌ Пользователь не найден")
            return

        user_id = user.id
        print(f"👤 Очистка данных пользователя: {user.email}")

        # Удаляем все документы
        print("\n🧹 Удаление документов из PostgreSQL...")
        result = await db.execute(delete(Source).where(Source.user_id == user_id))
        sources_count = result.rowcount
        print(f"   ✅ Удалено документов: {sources_count}")

        # Удаляем все папки
        print("\n🧹 Удаление папок из PostgreSQL...")
        result = await db.execute(delete(Folder).where(Folder.user_id == user_id))
        folders_count = result.rowcount
        print(f"   ✅ Удалено папок: {folders_count}")

        # Сохраняем изменения
        await db.commit()

    await close_db()


async def main():
    """Главная функция"""
    print("🚀 Очистка базы данных PostgreSQL...")
    print("⚠️  ВНИМАНИЕ: Это удалит все документы и папки!")
    print("ℹ️  ChromaDB не будет затронут")
    print()

    # Запрос подтверждения
    response = input("Продолжить? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("❌ Отменено")
        return

    await cleanup_database()
    print("\n✨ Готово! PostgreSQL очищен (ChromaDB не тронут).")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. Файл backend/.env существует")
        sys.exit(1)
