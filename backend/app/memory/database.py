"""Database engine setup, connection management, and lifecycle initialization."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.logging import get_logger
from app.memory.models import Base

logger = get_logger("memory.database")


def is_file_based_sqlite(database_url: str) -> bool:
    """Determine whether the SQLite database URL targets a persistent filesystem file."""
    if not database_url.startswith(("sqlite://", "sqlite+aiosqlite://")):
        return False

    cleaned = database_url
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]

    return cleaned not in (":memory:", "", "?cache=shared")


def ensure_db_directory(database_url: str) -> None:
    """Ensure directory exists on disk for SQLite file storage.

    Safely handles in-memory and absolute/relative path URLs.
    """
    if not is_file_based_sqlite(database_url):
        return

    cleaned = database_url
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]

    db_path = Path(cleaned)
    db_dir = db_path.parent
    if str(db_dir) and not db_dir.exists():
        logger.info("Creating directory for persistent database at: %s", db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)


def create_database_engine(
    database_url: str,
    echo: bool = False,
    busy_timeout_ms: int = 5000,
) -> AsyncEngine:
    """Create and configure an asynchronous SQLAlchemy engine.

    Enforces SQLite foreign key constraints, busy timeout, and WAL mode
    for file-based databases to maximize concurrent read/write throughput.
    """
    ensure_db_directory(database_url)
    is_file_db = is_file_based_sqlite(database_url)

    engine = create_async_engine(
        database_url,
        echo=echo,
        future=True,
    )

    # Configure per-connection pragmas for concurrency, data integrity, and performance
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        if is_file_db:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def get_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async sessionmaker bound to the given engine."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def init_db(engine: AsyncEngine) -> None:
    """Initialize database tables for persistent conversation memory."""
    logger.info("Verifying and initializing memory database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Memory database initialization complete")


@asynccontextmanager
async def get_db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Context manager yielding an AsyncSession with automatic commit and rollback."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
