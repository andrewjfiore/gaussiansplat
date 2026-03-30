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
    error TEXT
);
"""

_DB_MIGRATIONS = [
    "ALTER TABLE projects ADD COLUMN video_type TEXT DEFAULT 'standard'",
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


async def init_db():
    settings.ensure_dirs()
    db = await get_db()
    try:
        await db.executescript(_DB_INIT)
        await db.commit()
        await run_migrations(db)
    finally:
        await db.close()
