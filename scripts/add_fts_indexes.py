#!/usr/bin/env python3
"""Legacy one-off PostgreSQL FTS DDL helper.

Current ``backend.db.init_db`` already owns equivalent tsvector columns, triggers,
backfill and indexes. This script mutates the database selected by backend
settings, drops/recreates triggers and has no migration ledger or downgrade.
Use only after comparing its SQL with ``backend/db.py`` and backing up the target.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text
from settings import settings


async def add_fts_indexes():
    """Add FTS indexes to sources table"""

    pg_settings = settings.POSTGRES
    DATABASE_URL = f"postgresql+asyncpg://{pg_settings.USER}:{pg_settings.PASSWORD}@{pg_settings.HOST}:{pg_settings.PORT}/{pg_settings.DATABASE}"

    engine = create_async_engine(DATABASE_URL, echo=True)

    print("🔍 Adding Full-Text Search indexes to sources table...")

    try:
        async with engine.begin() as conn:
            # Add tsvector columns if they don't exist
            print("📝 Adding tsvector columns...")
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'sources' AND column_name = 'name_tsvector'
                    ) THEN
                        ALTER TABLE sources ADD COLUMN name_tsvector tsvector;
                        RAISE NOTICE 'Added name_tsvector column';
                    ELSE
                        RAISE NOTICE 'name_tsvector column already exists';
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'sources' AND column_name = 'content_tsvector'
                    ) THEN
                        ALTER TABLE sources ADD COLUMN content_tsvector tsvector;
                        RAISE NOTICE 'Added content_tsvector column';
                    ELSE
                        RAISE NOTICE 'content_tsvector column already exists';
                    END IF;
                END $$;
            """))

            # Create GIN indexes if they don't exist
            print("🔨 Creating GIN indexes...")
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_sources_name_fts ON sources USING GIN(name_tsvector)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_sources_content_fts ON sources USING GIN(content_tsvector)"
            ))

            # Create or replace triggers to automatically update tsvector columns
            print("⚡ Creating trigger functions...")
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION sources_name_tsvector_update() RETURNS trigger AS $$
                BEGIN
                    NEW.name_tsvector := to_tsvector('english', COALESCE(regexp_replace(NEW.name, '[._-]', ' ', 'g'), ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """))

            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION sources_content_tsvector_update() RETURNS trigger AS $$
                BEGIN
                    NEW.content_tsvector := to_tsvector('english', COALESCE(NEW.content, ''));
                    RETURN NEW;
                END
                $$ LANGUAGE plpgsql
            """))

            print("⚡ Creating triggers...")
            await conn.execute(text("DROP TRIGGER IF EXISTS sources_name_tsvector_trigger ON sources"))
            await conn.execute(text("""
                CREATE TRIGGER sources_name_tsvector_trigger
                    BEFORE INSERT OR UPDATE ON sources
                    FOR EACH ROW EXECUTE FUNCTION sources_name_tsvector_update()
            """))

            await conn.execute(text("DROP TRIGGER IF EXISTS sources_content_tsvector_trigger ON sources"))
            await conn.execute(text("""
                CREATE TRIGGER sources_content_tsvector_trigger
                    BEFORE INSERT OR UPDATE ON sources
                    FOR EACH ROW EXECUTE FUNCTION sources_content_tsvector_update()
            """))

            # Update existing rows
            print("🔄 Updating existing documents with FTS vectors...")
            result = await conn.execute(text("""
                UPDATE sources
                SET name_tsvector = to_tsvector('english', COALESCE(name, ''))
                WHERE name_tsvector IS NULL;
            """))
            print(f"   Updated {result.rowcount} documents (name)")

            result = await conn.execute(text("""
                UPDATE sources
                SET content_tsvector = to_tsvector('english', COALESCE(content, ''))
                WHERE content_tsvector IS NULL;
            """))
            print(f"   Updated {result.rowcount} documents (content)")

        print("\n✅ Full-Text Search indexes added successfully!")
        print("📚 You can now search documents by name or content with fast performance.")

    except Exception as e:
        print(f"\n❌ Error adding FTS indexes: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(add_fts_indexes())
