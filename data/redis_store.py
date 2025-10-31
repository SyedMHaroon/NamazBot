# data/redis_store.py
import os
import json
from typing import Any, Optional, Dict

import redis.asyncio as redis   # pip install "redis>=5"

# Example:
# REDIS_URL=rediss://default:password@host:port/0
# You can also append "?ssl_cert_reqs=none" if your env has custom CAs.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()

# Optional override for TLS cert policy when building from granular envs
REDIS_SSL_CERT_REQS = os.getenv("REDIS_SSL_CERT_REQS", "").strip().lower()
if REDIS_SSL_CERT_REQS not in {"", "none", "required"}:
    REDIS_SSL_CERT_REQS = ""

# Session & buffers TTLs (tweak if needed)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))  # 7 days
BUFFER_TTL_SECONDS  = int(os.getenv("BUFFER_TTL_SECONDS",  "604800"))  # 7 days
BUFFER_MAXLEN       = int(os.getenv("BUFFER_MAXLEN", "40"))            # keep last 40 msgs

# Single global client (connection pooling handled by library)
_redis: Optional[redis.Redis] = None

def get_redis() -> redis.Redis:
    """
    Returns a singleton async Redis client using REDIS_URL.
    Supports redis:// and rediss:// (TLS) URLs with username/password (e.g., Redis Cloud).
    """
    global _redis
    if _redis is None:
        kwargs = dict(decode_responses=True)
        # If user asked for explicit SSL cert policy and URL didn't specify it,
        # redis-py will honor query params in the URL (preferred). Otherwise we
        # can still pass 'ssl_cert_reqs' via from_url's connection kwargs if needed.
        if REDIS_SSL_CERT_REQS in {"none", "required"} and "ssl_cert_reqs=" not in REDIS_URL:
            # Map to what redis-py expects
            kwargs["ssl_cert_reqs"] = None if REDIS_SSL_CERT_REQS == "none" else "required"
        _redis = redis.from_url(REDIS_URL, **kwargs)
    return _redis

# ---------- Recent message de-dup (idempotency) ----------
# Key format: recent:{wa_id}
# Returns True if we've seen it recently; False otherwise (and stores it).
async def already_seen(wa_id: str, msg_id: str, ttl_seconds: int = 3600) -> bool:
    if not msg_id:
        return False
    r = get_redis()
    key = f"recent:{wa_id}"
    try:
        added = await r.sadd(key, msg_id)  # 1 if new, 0 if exists
        await r.expire(key, ttl_seconds)
        return added == 0
    except Exception as e:
        # Avoid breaking the webhook path if Redis is transiently unavailable
        print(f"[WARN] redis_already_seen failed: {e}")
        return False

# ---------- Simple JSON get/set ----------
async def set_json(key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
    r = get_redis()
    data = json.dumps(value, ensure_ascii=False)
    if ttl_seconds:
        await r.set(key, data, ex=ttl_seconds)
    else:
        await r.set(key, data)

async def get_json(key: str) -> Optional[Any]:
    r = get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

# ---------- Small TTL cache helpers (strings) ----------
async def cache_get(key: str) -> Optional[str]:
    r = get_redis()
    return await r.get(key)

async def cache_set(key: str, value: str, ttl_seconds: int) -> None:
    r = get_redis()
    await r.set(key, value, ex=ttl_seconds)

# ---------- Session helpers used by session_store.py ----------
# Key: sess:{wa_id}  -> JSON dict { "profile": {...}, ... }
async def get_session(wa_id: str) -> Dict[str, Any]:
    r = get_redis()
    key = f"sess:{wa_id}"
    raw = await r.get(key)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}

async def set_session(wa_id: str, session_obj: Dict[str, Any]) -> None:
    r = get_redis()
    key = f"sess:{wa_id}"
    await r.set(key, json.dumps(session_obj, ensure_ascii=False), ex=SESSION_TTL_SECONDS)

# ---------- Lightweight rolling buffer of last messages ----------
# List key: buf:{wa_id} -> each entry is a JSON string {"role": "...", "text": "..."}
async def add_buffered_message(wa_id: str, role: str, text: str) -> None:
    """
    Push (role, text) into a Redis list and trim to BUFFER_MAXLEN.
    Keeps the **last N** items using negative indices.
    """
    if not text:
        return
    r = get_redis()
    key = f"buf:{wa_id}"
    item = json.dumps({"role": role, "text": text}, ensure_ascii=False)
    # Append to tail; then trim to last N items
    await r.rpush(key, item)
    await r.ltrim(key, -BUFFER_MAXLEN, -1)  # <-- keep last N
    await r.expire(key, BUFFER_TTL_SECONDS)

# Optional: health check
async def ping() -> bool:
    r = get_redis()
    try:
        return (await r.ping()) is True
    except Exception:
        return False
