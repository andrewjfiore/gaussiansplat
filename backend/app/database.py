import aiosqlite
from pathlib import Path
from .config import settings

_DB_INIT = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    step TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL,
    video_filename TEXT,
    frame_count INTEGER DEFAULT 0,
    sfm_points INTEGER DEFAULT 0,
    training_iterations INTEGER DEFAULT 0,
    has_output INTEGER DEFAULT 0,
    error TEXT
);
"""


async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(str(settings.db_path))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    return db


async def init_db():
    settings.ensure_dirs()
    db = await get_db()
    try:
        await db.executescript(_DB_INIT)
        await db.commit()
    finally:
        await db.close()
