#!/usr/bin/env python3
"""
Скрипт для создания тестового пользователя в БД
"""
import sys
import os
import asyncio
import bcrypt
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env файл из backend
backend_dir = Path(__file__).parent.parent / "backend"
env_path = backend_dir / ".env"
load_dotenv(env_path)

# Добавляем корневую директорию в PATH
sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from db import init_db, close_db, get_db, User


def hash_password(password: str) -> str:
    """Хешировать пароль с помощью bcrypt"""
    # Конвертируем пароль в байты
    password_bytes = password.encode('utf-8')
    # Генерируем соль и хешируем
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Возвращаем как строку
    return hashed.decode('utf-8')


async def create_test_user():
    """Создать тестового пользователя"""
    await init_db()

    async for db in get_db():
        # Проверяем, существует ли пользователь
        result = await db.execute(
            select(User).where(User.email == "user@example.com")
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print("✅ Пользователь user@example.com уже существует")
            print(f"   ID: {existing_user.id}")
            print(f"   Username: {existing_user.username}")
            print(f"   Email: {existing_user.email}")
            return

        # Создаем нового пользователя
        hashed_password = hash_password("password")

        new_user = User(
            email="user@example.com",
            username="testuser",
            password_hash=hashed_password,
            is_active=True
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        print("✅ Тестовый пользователь создан успешно!")
        print(f"   ID: {new_user.id}")
        print(f"   Email: user@example.com")
        print(f"   Password: password")
        print(f"   Username: testuser")

    await close_db()


if __name__ == "__main__":
    print("🔧 Создание тестового пользователя...")
    try:
        asyncio.run(create_test_user())
        print("\n✨ Готово! Теперь можете войти с учетными данными:")
        print("   Email: user@example.com")
        print("   Password: password")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("  1. PostgreSQL запущен")
        print("  2. Файл backend/.env существует и содержит настройки POSTGRES_*")
        sys.exit(1)
