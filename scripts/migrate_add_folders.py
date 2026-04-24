#!/usr/bin/env python3
"""
Миграция: добавление таблицы folders и поля folder_id в sources
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

from sqlalchemy import text
from db import engine


async def migrate():
    """Добавить таблицу folders и поле folder_id"""
    print("🚀 Миграция базы данных: добавление поддержки папок...")

    async with engine.begin() as conn:
        # Проверяем, существует ли таблица folders
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'folders'
            );
        """))
        folders_exists = result.scalar()

        if not folders_exists:
            print("\n📦 Создание таблицы folders...")
            await conn.execute(text("""
                CREATE TABLE folders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    path VARCHAR(1000) NOT NULL,
                    parent_id INTEGER,
                    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_id) REFERENCES folders(id) ON DELETE CASCADE
                );
            """))
            await conn.execute(text("""
                CREATE INDEX ix_folders_id ON folders(id);
            """))
            await conn.execute(text("""
                CREATE INDEX ix_folders_user_id ON folders(user_id);
            """))
            await conn.execute(text("""
                CREATE INDEX ix_folders_parent_id ON folders(parent_id);
            """))
            print("   ✅ Таблица folders создана")
        else:
            print("\n⏭️  Таблица folders уже существует")

        # Проверяем, существует ли поле folder_id в sources
        result = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name = 'sources'
                AND column_name = 'folder_id'
            );
        """))
        folder_id_exists = result.scalar()

        if not folder_id_exists:
            print("\n📦 Добавление поля folder_id в таблицу sources...")
            await conn.execute(text("""
                ALTER TABLE sources
                ADD COLUMN folder_id INTEGER;
            """))
            await conn.execute(text("""
                ALTER TABLE sources
                ADD CONSTRAINT sources_folder_id_fkey
                FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE CASCADE;
            """))
            await conn.execute(text("""
                CREATE INDEX ix_sources_folder_id ON sources(folder_id);
            """))
            print("   ✅ Поле folder_id добавлено в sources")
        else:
            print("\n⏭️  Поле folder_id уже существует в sources")

    await engine.dispose()
    print("\n✨ Готово! База данных обновлена.")
    print("\nТеперь можете запустить:")
    print("  uv run scripts/load_documents_with_folders.py")


if __name__ == "__main__":
    try:
        asyncio.run(migrate())
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. Файл backend/.env существует")
        sys.exit(1)
