# data/db_postgres.py
import os
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, Integer, BigInteger, DateTime, ForeignKey,
    func, text, bindparam, select
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv

load_dotenv()

# ---------- Config (force async driver if a sync DSN was pasted) ----------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/muslim_bot",
)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

Base = declarative_base()

# ---------- Models ----------
class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    # WhatsApp number (or any stable chat id). Keep string to preserve leading '+'
    wa_id = Column(String(64), nullable=False, unique=True, index=True)

    name = Column(String(120))
    email = Column(String(190))
    city = Column(String(120))
    country = Column(String(120))
    tz = Column(String(120))          # e.g., "Asia/Karachi"
    lang = Column(String(5))          # 'en' | 'ar'
    pipedream_connection_id = Column(String(255), nullable=True)  # Pipedream connection ID for Google Calendar

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    messages = relationship("Message", back_populates="user", lazy="raise", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(16), nullable=False)   # 'user' | 'assistant' | 'system'
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    user = relationship("User", back_populates="messages", lazy="raise")


# ---------- Engine / Session ----------
# Neon “-pooler” works well with a small pool.
engine = create_async_engine(
    DATABASE_URL,
    echo=False,              # set True to debug SQL
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=2,
    future=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# ---------- Public API ----------
async def init_db() -> None:
    """Create tables if they don't exist; quick connectivity check."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("SELECT 1"))


async def get_user(wa_id: str) -> Optional[Dict[str, Any]]:
    """Return a user row as a dict (or None)."""
    async with SessionLocal() as session:
        q = await session.execute(
            text(
                """
                SELECT id, wa_id, name, email, city, country, tz, lang, pipedream_connection_id, created_at, updated_at
                FROM users WHERE wa_id = :wa_id
                """
            ),
            {"wa_id": wa_id},
        )
        row = q.mappings().first()
        return dict(row) if row else None


async def upsert_user_profile(
    wa_id: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    tz: Optional[str] = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Insert or update a user's profile fields (only non-None fields overwrite).
    Returns the full user row.
    """
    async with SessionLocal() as session:
        user_tbl = User.__table__
        insert_stmt = pg_insert(user_tbl).values(
            wa_id=wa_id, name=name, email=email, city=city, country=country, tz=tz, lang=lang
        )
        update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[user_tbl.c.wa_id],
            set_={
                "name": func.coalesce(insert_stmt.excluded.name, user_tbl.c.name),
                "email": func.coalesce(insert_stmt.excluded.email, user_tbl.c.email),
                "city": func.coalesce(insert_stmt.excluded.city, user_tbl.c.city),
                "country": func.coalesce(insert_stmt.excluded.country, user_tbl.c.country),
                "tz": func.coalesce(insert_stmt.excluded.tz, user_tbl.c.tz),
                "lang": func.coalesce(insert_stmt.excluded.lang, user_tbl.c.lang),
                "updated_at": func.now(),
            },
        ).returning(
            user_tbl.c.id, user_tbl.c.wa_id, user_tbl.c.name, user_tbl.c.email,
            user_tbl.c.city, user_tbl.c.country, user_tbl.c.tz, user_tbl.c.lang,
            user_tbl.c.created_at, user_tbl.c.updated_at
        )

        result = await session.execute(update_stmt)
        row = result.mappings().first()
        await session.commit()
        return dict(row)


async def _ensure_user(session: AsyncSession, wa_id: str) -> int:
    """Get user id; create if missing. Returns user_id."""
    user_tbl = User.__table__
    stmt = pg_insert(user_tbl).values(wa_id=wa_id).on_conflict_do_nothing().returning(user_tbl.c.id)
    res = await session.execute(stmt)
    user_id = res.scalar_one_or_none()
    if user_id is not None:
        return user_id
    res2 = await session.execute(select(User.id).where(User.wa_id == wa_id))
    return res2.scalar_one()


async def append_message(wa_id: str, role: str, text_: str) -> None:
    """Store a message row. Creates the user if missing (without extra profile info)."""
    async with SessionLocal() as session:
        user_id = await _ensure_user(session, wa_id)
        await session.execute(
            text("INSERT INTO messages (user_id, role, text) VALUES (:user_id, :role, :text)"),
            {"user_id": user_id, "role": role, "text": text_},
        )
        await session.commit()


async def fetch_last_messages(wa_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return the most recent N messages for this user, in **chronological** order.
    Each row: {role, text, created_at}
    """
    async with SessionLocal() as session:
        q = await session.execute(
            text(
                """
                SELECT m.role, m.text, m.created_at
                FROM messages m
                JOIN users u ON u.id = m.user_id
                WHERE u.wa_id = :wa_id
                ORDER BY m.created_at DESC
                LIMIT :limit
                """
            ).bindparams(
                bindparam("wa_id", wa_id),
                bindparam("limit", limit, type_=Integer),
            )
        )
        rows = q.mappings().all()
        return [dict(r) for r in reversed(rows)]  # oldest → newest


async def trim_messages_to(wa_id: str, keep: int = 10) -> int:
    """
    Keep only the newest `keep` messages for this user. Returns rows deleted.
    """
    async with SessionLocal() as session:
        res = await session.execute(select(User.id).where(User.wa_id == wa_id))
        user_id = res.scalar_one_or_none()
        if user_id is None:
            return 0

        deleted = await session.execute(
            text(
                """
                WITH to_keep AS (
                    SELECT id
                    FROM messages
                    WHERE user_id = :uid
                    ORDER BY created_at DESC
                    LIMIT :keep
                )
                DELETE FROM messages
                WHERE user_id = :uid
                  AND id NOT IN (SELECT id FROM to_keep)
                """
            ).bindparams(
                bindparam("uid", user_id),
                bindparam("keep", keep, type_=Integer),
            )
        )
        await session.commit()
        return deleted.rowcount if (deleted.rowcount or 0) > 0 else 0


# ---------- Pipedream Connection ID management ----------
async def get_pipedream_connection_id(wa_id: str) -> Optional[str]:
    """Get Pipedream connection ID for a user."""
    async with SessionLocal() as session:
        res = await session.execute(
            select(User.pipedream_connection_id).where(User.wa_id == wa_id)
        )
        return res.scalar_one_or_none()


async def set_pipedream_connection_id(wa_id: str, connection_id: str) -> None:
    """Set Pipedream connection ID for a user."""
    async with SessionLocal() as session:
        await session.execute(
            text("UPDATE users SET pipedream_connection_id = :conn_id, updated_at = NOW() WHERE wa_id = :wa_id"),
            {"conn_id": connection_id, "wa_id": wa_id}
        )
        await session.commit()


async def clear_pipedream_connection_id(wa_id: str) -> None:
    """Clear/disconnect Pipedream connection ID for a user."""
    async with SessionLocal() as session:
        await session.execute(
            text("UPDATE users SET pipedream_connection_id = NULL, updated_at = NOW() WHERE wa_id = :wa_id"),
            {"wa_id": wa_id}
        )
        await session.commit()
        print(f"[PIPEDREAM] Cleared connection ID for wa_id: {wa_id}")


# ---------- Back-compat aliases (so older imports keep working) ----------
bootstrap = init_db
get_user_by_phone = get_user
upsert_user = upsert_user_profile
get_last_messages = fetch_last_messages
