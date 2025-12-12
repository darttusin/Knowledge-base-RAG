from datetime import datetime
from typing import AsyncGenerator

from settings import settings
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RequestHistory(Base):
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, index=True)
    tg_user_id = Column(Integer, nullable=True, index=True)
    request_type = Column(String(20), nullable=False)
    request_data = Column(JSON, nullable=False)
    response_data = Column(JSON, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    input_length = Column(Integer, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)


pg_settings = settings.POSTGRES
DATABASE_URL = f"postgresql+asyncpg://{pg_settings.USER}:{pg_settings.PASSWORD}@{pg_settings.HOST}:{pg_settings.PORT}/{pg_settings.DATABASE}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession]:
    async with AsyncSession(engine) as s:
        yield s
