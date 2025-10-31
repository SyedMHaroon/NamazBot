# session_store.py
from typing import Dict, Any, List, Tuple

# Postgres data layer (matches data/db_postgres.py)
from data.db_postgres import (
    init_db,
    get_user,                 # get_user(wa_id: str) -> Optional[Dict[str, Any]]
    upsert_user_profile,      # upsert_user_profile(wa_id, name, email, city, country, tz, lang) -> Dict
    append_message,           # append_message(wa_id, role, text_) -> None
    fetch_last_messages,      # fetch_last_messages(wa_id, limit) -> List[Dict]
)

# Redis cache/session helpers (as per your redis_store.py)
from data.redis_store import get_session, set_session, add_buffered_message


# ---------- Bootstrap ----------
async def init_data_layer() -> None:
    """Initialize underlying stores (e.g., create tables)."""
    await init_db()


# ---------- Profile ----------
async def get_profile(user_phone: str) -> Dict[str, Any]:
    """
    Merge durable profile (Postgres) with ephemeral session (Redis).

    Returns a dict shaped like:
      {
        "name": str, "email": str, "city": str, "country": str,
        "tz": str, "lang": "en"|"ar",
        ...plus any transient flags from Redis (e.g., _await, _confirm_loc)
      }
    """
    # Durable profile from Postgres
    user_row = await get_user(user_phone)  # dict or None

    # Ephemeral session from Redis
    sess = await get_session(user_phone)   # dict (possibly empty)
    profile = dict(sess.get("profile", {}))  # shallow copy

    if user_row:
        profile.setdefault("name",    user_row.get("name") or "")
        profile.setdefault("email",   user_row.get("email") or "")
        profile.setdefault("city",    user_row.get("city") or "")
        profile.setdefault("country", user_row.get("country") or "")
        if user_row.get("tz"):
            profile.setdefault("tz",  user_row.get("tz"))
        profile.setdefault("lang",    (user_row.get("lang") or "en"))

    return profile


async def set_profile(user_phone: str, profile: Dict[str, Any]) -> None:
    """
    Persist normalized profile fields to Postgres and cache the full profile in Redis.
    Transient flags remain only in Redis.
    """
    # Push structured fields into Postgres (durable)
    await upsert_user_profile(
        wa_id=user_phone,
        name=profile.get("name"),
        email=profile.get("email"),
        city=profile.get("city"),
        country=profile.get("country"),
        tz=profile.get("tz"),
        lang=profile.get("lang"),
    )

    # Cache the whole profile (including transient flags) in Redis
    sess = await get_session(user_phone)
    sess["profile"] = profile
    await set_session(user_phone, sess)


# ---------- Conversation log ----------
async def add_turn(user_phone: str, user_text: str, bot_text: str, **_) -> None:
    """
    Append a user→assistant turn to Postgres, and also keep a short buffer in Redis for fast access.
    Accepts extra kwargs (e.g., lang=...), which are ignored to avoid signature errors.
    """
    # Durable message log
    await append_message(user_phone, "user", user_text)
    await append_message(user_phone, "assistant", bot_text)

    # Lightweight rolling buffer in Redis
    await add_buffered_message(user_phone, "user", user_text)
    await add_buffered_message(user_phone, "assistant", bot_text)


async def fetch_context(user_phone: str, limit: int = 10) -> List[Tuple[str, str]]:
    """
    Return last N (role, text) messages in chronological order.
    Uses Postgres as the source of truth.
    """
    rows = await fetch_last_messages(user_phone, limit=limit)
    # fetch_last_messages returns chronological list with keys: role, text, created_at
    return [(r["role"], r["text"]) for r in rows]
