#!/usr/bin/env python3
"""
Migration script to add sources column to messages table.
"""
import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from db import engine  # noqa: E402


async def migrate() -> None:
    """Add sources column to messages table."""
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(
            text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'messages' AND column_name = 'sources'
            """)
        )

        if result.fetchone():
            return

        # Add sources column
        await conn.execute(
            text("ALTER TABLE messages ADD COLUMN sources TEXT")
        )
if __name__ == "__main__":
    asyncio.run(migrate())
