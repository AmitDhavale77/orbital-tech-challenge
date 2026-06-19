from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from takehome.config import settings
from takehome.db.models import Base


def _split_url() -> tuple[str, str]:
    """Return (base, db_name) from the configured database URL."""
    base, _, name = settings.database_url.rpartition("/")
    return base, name


def _test_db_url() -> str:
    base, name = _split_url()
    return f"{base}/{name}_test"


async def _ensure_test_database() -> None:
    """Create the dedicated test database if it does not yet exist."""
    base, name = _split_url()
    admin_dsn = f"{base}/postgres".replace("postgresql+asyncpg", "postgresql")
    test_name = f"{name}_test"
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_name
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{test_name}"')
    finally:
        await conn.close()


@pytest.fixture
async def sessionmaker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A sessionmaker bound to a freshly-truncated test database.

    Schema is created from the SQLAlchemy models (decoupled from Alembic), so
    tests always run against the current models.
    """
    await _ensure_test_database()
    engine = create_async_engine(_test_db_url(), echo=False)
    # Reset the schema so it always matches the current models — robust to tables
    # dropped from the models across tickets — and give each test empty tables.
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        yield maker
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
async def stub_card_agent() -> AsyncIterator[None]:
    """Stub the Haiku card model so uploads in tests don't call the real LLM."""
    from pydantic_ai.models.test import TestModel

    from takehome.services.cards import card_agent

    with card_agent.override(model=TestModel()):
        yield


@pytest.fixture
async def db_session(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with sessionmaker() as session:
        yield session


@pytest.fixture
async def client(
    sessionmaker: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """An ASGI client wired to the test database.

    Overrides the request-scoped `get_session` dependency *and* patches the
    module-level session factory the SSE generator opens for its post-stream
    save, so the whole request path hits the test database.
    """
    import takehome.db.session as db_session_module
    from takehome.db.session import get_session
    from takehome.web.app import app

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with sessionmaker() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    monkeypatch.setattr(db_session_module, "async_session", sessionmaker)

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
