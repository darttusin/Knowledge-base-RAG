#!/usr/bin/env python3
"""Destructive legacy reset that drops every ORM-managed database table.

Interactive confirmation does not make this recoverable: all PostgreSQL users,
dialogues, messages, folders and sources are lost, while ChromaDB remains stale.
The script calls ``Base.metadata.create_all`` directly and does not run the full
``init_db`` FTS/upgrade DDL afterwards. Never use it on shared/prod data.
"""
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл из backend
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"
load_dotenv(env_path)

# Добавляем корневую директорию в PATH
sys.path.insert(0, str(backend_dir))

from db import engine, Base


async def recreate_tables():
    """Пересоздать все таблицы"""
    print("🚀 Пересоздание таблиц базы данных...")
    print("⚠️  ВНИМАНИЕ: Это удалит все данные!")
    print()

    # Запрос подтверждения
    response = input("Продолжить? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("❌ Отменено")
        return

    print("\n🗑️  Удаление старых таблиц...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("   ✅ Старые таблицы удалены")

    print("\n📦 Создание новых таблиц...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("   ✅ Новые таблицы созданы")

    print("\n✨ Готово! Таблицы пересозданы с новой структурой.")
    print("\nТеперь запустите:")
    print("  1. uv run scripts/create_test_user.py")
    print("  2. uv run scripts/load_documents_with_folders.py")

    await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(recreate_tables())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. Файл backend/.env существует")
        sys.exit(1)
