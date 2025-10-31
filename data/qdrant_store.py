# data/qdrant_store.py
import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional

import httpx

# NOTE: our embeddings client is sync; we wrap it for async usage
from .embeddings import embed_query as _embed_query_sync, embed_many as _embed_many_sync, EMBED_DIM

# -----------------------------
# Env/config
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "conv_memory")

# Distance: "Cosine" | "Dot" | "Euclid"
# For BGE models, keep Cosine + L2-normalized vectors (handled in embeddings.py)
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")

# If you know the vector size, set it; otherwise we’ll pull from EMBED_DIM
QDRANT_VECTOR_SIZE_ENV = os.getenv("QDRANT_VECTOR_SIZE", "")
QDRANT_VECTOR_SIZE: Optional[int] = (
    int(QDRANT_VECTOR_SIZE_ENV) if QDRANT_VECTOR_SIZE_ENV.isdigit() else None
)

# -----------------------------
# HTTP client
# -----------------------------
_client: Optional[httpx.AsyncClient] = None

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h

def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=5.0),
            headers=_headers(),
        )
    return _client

async def close_http():
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        finally:
            _client = None

# -----------------------------
# Embedding wrappers (sync -> async)
# -----------------------------
async def _embed_one(text: str) -> List[float]:
    return await asyncio.to_thread(_embed_query_sync, text)

async def _embed_batch(texts: List[str]) -> List[List[float]]:
    # Uses the batch endpoint in the sync client; one thread hop
    return await asyncio.to_thread(_embed_many_sync, texts)

# -----------------------------
# Collection management
# -----------------------------
async def _collection_exists() -> bool:
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}"
    r = await _http().get(url)
    return r.status_code == 200

async def _detect_vector_size() -> int:
    """Use EMBED_DIM if provided; else embed a probe string to detect."""
    global QDRANT_VECTOR_SIZE
    if QDRANT_VECTOR_SIZE:
        return QDRANT_VECTOR_SIZE
    if EMBED_DIM:  # from embeddings.py defaults / .env
        QDRANT_VECTOR_SIZE = EMBED_DIM
        return QDRANT_VECTOR_SIZE
    v = await _embed_one("probe")
    if not v:
        raise RuntimeError("Failed to detect embedding vector size (empty vector).")
    QDRANT_VECTOR_SIZE = len(v)
    return QDRANT_VECTOR_SIZE

async def _create_user_id_index() -> None:
    """Create a payload index on user_id for faster filtered search."""
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/index"
    body = {
        "field_name": "user_id",
        "field_schema": "keyword"  # string exact-match index
    }
    # Qdrant returns 200 on success, 409 if already exists — both are fine
    r = await _http().put(url, json=body)
    if r.status_code not in (200, 409):
        r.raise_for_status()

async def ensure_collection(recreate: bool = False) -> None:
    """
    Ensure collection exists with the right vector size and distance.
    Set recreate=True if you want to force-drop and recreate (careful: data loss).
    """
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}"

    if recreate:
        await _http().delete(url)  # ignore errors; best effort

    if await _collection_exists():
        # Still ensure index exists
        await _create_user_id_index()
        return

    dim = await _detect_vector_size()
    payload = {
        "vectors": {
            "size": dim,
            "distance": QDRANT_DISTANCE
        }
        # You can also tune hnsw_config/quantization/optimizers here if needed
    }
    r = await _http().put(url, json=payload)
    r.raise_for_status()

    # Create index on user_id for faster filtering
    await _create_user_id_index()

# -----------------------------
# Upsert messages
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)

async def add_message(
    user_id: str,
    text: str,
    role: str,                     # "user" | "assistant"
    ts_ms: Optional[int] = None,
    point_id: Optional[str] = None
) -> str:
    """
    Embed + upsert a single message for semantic memory.
    Returns the Qdrant point id.
    """
    await ensure_collection()
    vec = await _embed_one(text)
    if not vec:
        raise RuntimeError("Embedding failed: got empty vector.")

    pid = point_id or str(uuid.uuid4())
    ts = ts_ms or _now_ms()

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points"
    payload = {
        "points": [
            {
                "id": pid,
                "vector": vec,
                "payload": {
                    "user_id": user_id,
                    "role": role,
                    "text": text,
                    "ts": ts
                }
            }
        ]
    }
    r = await _http().put(url, json=payload)
    r.raise_for_status()
    return pid

async def add_messages_batch(
    user_id: str,
    messages: List[Dict[str, Any]],   # each: {"text": str, "role": "user"/"assistant", "ts": Optional[int], "id": Optional[str]}
) -> List[str]:
    """
    Batch embed + upsert. Efficient for backfilling.
    """
    await ensure_collection()

    texts = [m["text"] for m in messages]
    vectors = await _embed_batch(texts)

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points"
    points = []
    for vec, m in zip(vectors, messages):
        if not vec:
            # Skip silently or raise — here we skip
            continue
        pid = m.get("id") or str(uuid.uuid4())
        points.append({
            "id": pid,
            "vector": vec,
            "payload": {
                "user_id": user_id,
                "role": m.get("role", "user"),
                "text": m["text"],
                "ts": m.get("ts", _now_ms()),
            }
        })

    if not points:
        return []

    r = await _http().put(url, json={"points": points})
    r.raise_for_status()
    return [p["id"] for p in points]

# -----------------------------
# Semantic search (scoped to user)
# -----------------------------
async def search_similar(
    user_id: str,
    query_text: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Vector search within a user's memory.
    Returns list of {text, role, ts, score, id}
    """
    await ensure_collection()
    qvec = await _embed_one(query_text)
    if not qvec:
        return []

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"
    body: Dict[str, Any] = {
        "vector": qvec,
        "limit": int(top_k),
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [
                {"key": "user_id", "match": {"value": user_id}}
            ]
        }
    }
    if score_threshold is not None:
        body["score_threshold"] = float(score_threshold)

    r = await _http().post(url, json=body)
    r.raise_for_status()
    data = r.json()
    hits = data.get("result", []) or []

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.get("payload", {}) or {}
        results.append({
            "id": h.get("id"),
            "score": h.get("score"),
            "text": payload.get("text", ""),
            "role": payload.get("role", ""),
            "ts": payload.get("ts", 0),
        })
    return results
