# server.py
import os
import re
from typing import Any, Dict, List, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

# ---- Voice + WhatsApp helpers (your modules) ----
from voice_pipeline import stt_from_opus, tts_elevenlabs
from wa_client import send_whatsapp_text, send_whatsapp_audio
from formatting import normalize_for_tts

# ---- Bot turn handler (returns final-language text) ----
from main import handle_turn, start_scheduler, stop_scheduler, setup_digest_scheduler, setup_reminder_scheduler, setup_prayer_reminder_scheduler

# ---- Data layer: profiles + history (Postgres+Redis) ----
from session_store import (
    init_data_layer,
    get_profile,
    set_profile,
    add_turn,
    fetch_context,
)

# ---- Data layer: Redis idempotency ----
from data.redis_store import already_seen as redis_already_seen

# Digest schedule and deduplication settings
DIGEST_HOUR = int(os.getenv("DIGEST_HOUR"))
DIGEST_MINUTE = int(os.getenv("DIGEST_MINUTE"))
DIGEST_DEDUPE = os.getenv("DIGEST_DEDUPE")
# ---- Data layer: Qdrant memory ----
# TODO: Uncomment when embeddings API is working
# from data.qdrant_store import search_similar, add_message, ensure_collection, close_http as qdrant_close

VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN   = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID", "")
WABASE = "https://graph.facebook.com/v20.0"

app = FastAPI()

# ========== language helpers ==========
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

def detect_lang_from_text(s: str) -> str:
    """Return 'ar' if Arabic letters present, else 'en'."""
    if s and ARABIC_RE.search(s):
        return "ar"
    return "en"

def is_supported_lang(lang: str) -> bool:
    return (lang or "").lower() in ("en", "ar")

UNSUPPORTED_LANG_MSG = (
    "I currently support only English and Arabic. "
    "Please send your message in English or Arabic.\n\n"
    "أنا أدعم حاليًا اللغتين الإنجليزية والعربية فقط. "
    "من فضلك اكتب رسالتك باللغة الإنجليزية أو العربية."
)

# ========== app lifecycle ==========
@app.on_event("startup")
async def _startup():
    # Ensure Postgres tables exist, and Qdrant collection exists.
    await init_data_layer()
    # TODO: Uncomment when embeddings API is working
    # try:
    #     await ensure_collection()
    # except Exception as e:
    #     print("[WARN] ensure_collection failed (Qdrant):", e)
    
    # Initialize and start APScheduler
    await start_scheduler()
    
    # Setup digest scheduler
    # Set dedupe=False for testing (always sends), dedupe=True for production (once per day)
    setup_digest_scheduler(get_profile, send_whatsapp_text, hour=DIGEST_HOUR, minute=DIGEST_MINUTE, dedupe=DIGEST_DEDUPE)
    
    # Setup reminder tick scheduler (runs every minute)
    setup_reminder_scheduler(send_whatsapp_text)
    
    # Setup prayer reminder scheduler (runs every minute, sends reminders 10 min before each prayer)
    setup_prayer_reminder_scheduler(get_profile, send_whatsapp_text)

@app.on_event("shutdown")
async def _shutdown():
    try:
        await stop_scheduler()
    except Exception:
        pass
    # TODO: Uncomment when embeddings API is working
    # try:
    #     await qdrant_close()
    # except Exception:
    #     pass

# ========== health & verification ==========
@app.get("/")
async def health():
    return {"status": "ok"}

@app.get("/webhook")
async def whatsapp_verify(request: Request):
    params = dict(request.query_params)
    if (
        params.get("hub.mode") == "subscribe"
        and params.get("hub.verify_token") == VERIFY_TOKEN
        and params.get("hub.challenge")
    ):
        return PlainTextResponse(params["hub.challenge"])
    return PlainTextResponse("Verification failed", status_code=403)

# ========== helpers ==========
def extract_text(msg: Dict[str, Any]) -> str:
    mtype = msg.get("type")
    if mtype == "text":
        return (msg.get("text", {}) or {}).get("body", "").strip()
    if mtype == "interactive":
        i = msg.get("interactive", {}) or {}
        if i.get("type") == "button_reply":
            return (i.get("button_reply", {}) or {}).get("title", "").strip()
        if i.get("type") == "list_reply":
            return (i.get("list_reply", {}) or {}).get("title", "").strip()
    if "button" in msg:
        return (msg.get("button", {}) or {}).get("text", "").strip()
    return ""

async def fetch_whatsapp_media_bytes(media_id: str) -> bytes:
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_TOKEN missing.")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r1 = await client.get(f"{WABASE}/{media_id}", headers=headers)
        r1.raise_for_status()
        url = r1.json().get("url")
        if not url:
            raise RuntimeError("WhatsApp media URL not found.")
        r2 = await client.get(url, headers=headers)
        r2.raise_for_status()
        return r2.content

async def build_context(wa_from: str, user_text: str) -> Dict[str, Any]:
    """
    Compose the context dict for `handle_turn`:
      - short_history: last ~10 messages (role, text) from Postgres
      - semantic_snippets: top-k related lines from Qdrant
    """
    short_history: List[Tuple[str, str]] = await fetch_context(wa_from, limit=10)

    semantic_snippets: List[str] = []
    # TODO: Uncomment when embeddings API is working
    # try:
    #     hits = await search_similar(user_id=wa_from, query_text=user_text, top_k=5, score_threshold=0.35)
    #     semantic_snippets = [h.get("text", "") for h in hits if h.get("text")]
    # except Exception as e:
    #     # Don't fail the turn if Qdrant is down
    #     print("[WARN] Qdrant search_similar failed:", e)

    return {
        "short_history": short_history,         # [(role, text), ...] chronological
        "semantic_snippets": semantic_snippets  # [str, ...]
    }

async def persist_turn_everywhere(wa_from: str, user_text: str, bot_text: str, lang: str):
    """
    - Append (user, assistant) to Postgres
    - Add both messages to Qdrant for future recall
    """
    # Postgres
    await add_turn(wa_from, user_text, bot_text, lang=lang)

    # TODO: Uncomment when embeddings API is working
    # Qdrant (best-effort; don't raise)
    # try:
    #     await add_message(user_id=wa_from, text=user_text, role="user")
    #     if bot_text:
    #         await add_message(user_id=wa_from, text=bot_text, role="assistant")
    # except Exception as e:
    #     print("[WARN] Qdrant add_message failed:", e)

# ========== webhook (messages) ==========
@app.post("/webhook")
async def whatsapp_inbound(request: Request):
    """
    - Text  → detect 'ar'/'en' → build context → handle_turn → send text → persist to DB/Qdrant
    - Audio → STT(lang) → build context → handle_turn → TTS/audio → persist to DB/Qdrant
    """
    try:
        body: Dict[str, Any] = await request.json()

        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {}) or {}

                for msg in value.get("messages", []) or []:
                    wa_from = msg.get("from")
                    msg_id  = msg.get("id")
                    if not wa_from:
                        continue

                    # Global idempotency with Redis
                    try:
                        if await redis_already_seen(wa_from, msg_id or ""):
                            continue
                    except Exception as e:
                        # If Redis is briefly unavailable, proceed anyway
                        print("[WARN] redis_already_seen failed:", e)

                    mtype = msg.get("type")

                    # ---- TEXT MESSAGE ----
                    if mtype == "text" or mtype == "interactive" or "button" in msg:
                        text = extract_text(msg)
                        if not text:
                            continue

                        msg_lang = detect_lang_from_text(text)  # 'ar' or 'en'
                        if not is_supported_lang(msg_lang):
                            await send_whatsapp_text(wa_from, UNSUPPORTED_LANG_MSG)
                            continue

                        # Load + set profile language hint (final surface language handled in main.py)
                        profile = await get_profile(wa_from) or {}
                        temp_profile = dict(profile)
                        temp_profile["lang"] = msg_lang

                        # Build per-turn context (history + semantic recall)
                        ctx = await build_context(wa_from, text)

                        # Bot turn
                        reply_text, new_profile = await handle_turn(text, temp_profile, context=ctx, wa_id=wa_from)
                        await set_profile(wa_from, new_profile or profile)

                        # Normalize & send
                        reply_text_norm = normalize_for_tts(reply_text or "", msg_lang)
                        await send_whatsapp_text(wa_from, reply_text_norm or "…")

                        # Persist to DB + Qdrant
                        await persist_turn_everywhere(wa_from, text, reply_text_norm, lang=msg_lang)
                        continue

                    # ---- VOICE NOTE / AUDIO ----
                    if mtype == "audio":
                        audio_obj = (msg.get("audio") or {})
                        media_id = audio_obj.get("id")
                        if not media_id:
                            continue

                        try:
                            media_bytes = await fetch_whatsapp_media_bytes(media_id)

                            mime = (audio_obj.get("mime_type") or "").lower()
                            prefer_ext = ".ogg"
                            if "amr" in mime:   prefer_ext = ".amr"
                            elif "aac" in mime: prefer_ext = ".aac"
                            elif "mp4" in mime or "m4a" in mime: prefer_ext = ".m4a"
                            elif "mp3" in mime: prefer_ext = ".mp3"

                            stt_text, lang_from_stt = stt_from_opus(media_bytes, prefer_ext=prefer_ext)
                            if not stt_text:
                                await send_whatsapp_text(wa_from, "I couldn't understand the voice message. Please try again.")
                                continue

                            msg_lang = (lang_from_stt or "en").lower()
                            if not is_supported_lang(msg_lang):
                                await send_whatsapp_text(wa_from, UNSUPPORTED_LANG_MSG)
                                continue

                            profile = await get_profile(wa_from) or {}
                            temp_profile = dict(profile)
                            temp_profile["lang"] = msg_lang

                            # Context
                            ctx = await build_context(wa_from, stt_text)

                            # Bot turn
                            reply_text, new_profile = await handle_turn(stt_text, temp_profile, context=ctx, wa_id=wa_from)
                            await set_profile(wa_from, new_profile or profile)

                            # TTS or text fallback
                            reply_for_tts = normalize_for_tts(reply_text or "", msg_lang)
                            try:
                                mp3 = await tts_elevenlabs(reply_for_tts, msg_lang)
                                if mp3:
                                    await send_whatsapp_audio(wa_from, mp3)
                                else:
                                    await send_whatsapp_text(wa_from, reply_for_tts or "…")
                            except Exception as tts_err:
                                print("TTS error:", tts_err)
                                await send_whatsapp_text(wa_from, reply_for_tts or "…")

                            # Persist
                            await persist_turn_everywhere(wa_from, stt_text, reply_for_tts, lang=msg_lang)

                        except Exception as e:
                            print("Voice handling error:", e)
                            await send_whatsapp_text(
                                wa_from,
                                "There was an error processing your voice note. I'll reply in text for now."
                            )

    except Exception as e:
        print("Webhook processing error:", e)

    return JSONResponse({"status": "ok"})
