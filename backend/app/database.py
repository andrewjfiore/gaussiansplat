from contextlib import asynccontextmanager

import aiosqlite
from .config import settings

_DB_INIT = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    step TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL,
    video_filename TEXT,
    video_type TEXT DEFAULT 'standard',
    frame_count INTEGER DEFAULT 0,
    sfm_points INTEGER DEFAULT 0,
    training_iterations INTEGER DEFAULT 0,
    has_output INTEGER DEFAULT 0,
    error TEXT,
    cleanup_stats TEXT
);

CREATE TABLE IF NOT EXISTS project_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    video_index INTEGER NOT NULL,
    filename TEXT NOT NULL,
    video_type TEXT DEFAULT 'standard',
    uploaded_at TEXT NOT NULL,
    UNIQUE(project_id, video_index)
);
"""

_DB_MIGRATIONS = [
    "ALTER TABLE projects ADD COLUMN video_type TEXT DEFAULT 'standard'",
    "ALTER TABLE projects ADD COLUMN sfm_registered_images INTEGER DEFAULT 0",
    "ALTER TABLE projects ADD COLUMN sfm_reprojection_error REAL DEFAULT 0.0",
    "ALTER TABLE projects ADD COLUMN temporal_mode TEXT DEFAULT 'static'",
    "ALTER TABLE projects ADD COLUMN mask_keywords TEXT",
    "ALTER TABLE projects ADD COLUMN mask_count INTEGER DEFAULT 0",
    "ALTER TABLE projects ADD COLUMN cleanup_stats TEXT",
    "ALTER TABLE projects ADD COLUMN holefill_stats TEXT",
    "ALTER TABLE projects ADD COLUMN neural_refine_stats TEXT",
]


async def run_migrations(db):
    """Apply any missing schema migrations (idempotent)."""
    for sql in _DB_MIGRATIONS:
        try:
            await db.execute(sql)
            await db.commit()
        except Exception:
            pass  # Column already exists


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


@asynccontextmanager
async def db_session():
    """Async context manager: auto-commits on success, rolls back on error."""
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    finally:
        await db.close()


async def init_db():
    settings.ensure_dirs()
    db = await get_db()
    try:
        await db.executescript(_DB_INIT)
        await db.commit()
        await run_migrations(db)
    finally:
        await db.close()
