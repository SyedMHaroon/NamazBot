# server.py
import os
import re
from typing import Any, Dict, Deque
from collections import deque

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

# Voice + WhatsApp clients
from voice_pipeline import stt_from_opus, tts_elevenlabs
from wa_client import send_whatsapp_text, send_whatsapp_audio
from session_store import get_profile, set_profile
from formatting import normalize_for_tts

# LangGraph turn handler (main.py now handles final language of text)
from main import handle_turn

VERIFY_TOKEN     = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN   = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID  = os.getenv("PHONE_NUMBER_ID", "")

WABASE = "https://graph.facebook.com/v20.0"

app = FastAPI()

# ---------- language helpers ----------
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

# ---------- health & verification ----------
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

# ---------- helpers ----------
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

# ---------- de-dup (avoid processing the same message multiple times) ----------
_RECENT_MSG_IDS: Deque[str] = deque(maxlen=200)
_RECENT_MSG_SET = set()

def already_seen(msg_id: str) -> bool:
    if not msg_id:
        return False
    if msg_id in _RECENT_MSG_SET:
        return True
    _RECENT_MSG_SET.add(msg_id)
    _RECENT_MSG_IDS.append(msg_id)
    # prune if needed
    while len(_RECENT_MSG_SET) > _RECENT_MSG_IDS.maxlen:
        oldest = _RECENT_MSG_IDS.popleft()
        _RECENT_MSG_SET.discard(oldest)
    return False

# ---------- webhook (messages) ----------
@app.post("/webhook")
async def whatsapp_inbound(request: Request):
    """
    - Text  → detect 'ar'/'en' → handle_turn (main.py returns final language text) → send text
    - Audio → STT(lang) → handle_turn(lang hint) → TTS with same lang (no extra translate here)
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
                    if already_seen(msg_id):
                        continue

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

                        profile = get_profile(wa_from) or {}
                        temp_profile = dict(profile)
                        temp_profile["lang"] = msg_lang

                        reply_text, new_profile = await handle_turn(text, temp_profile)
                        set_profile(wa_from, new_profile or profile)

                        # No translation here — main.py returns the final language already.
                        print(f"[BOT REPLY] lang={msg_lang} → {reply_text}")

                        reply_text_norm = normalize_for_tts(reply_text or "", msg_lang)
                        await send_whatsapp_text(wa_from, reply_text_norm or "…")
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

                            profile = get_profile(wa_from) or {}
                            temp_profile = dict(profile)
                            temp_profile["lang"] = msg_lang

                            reply_text, new_profile = await handle_turn(stt_text, temp_profile)
                            set_profile(wa_from, new_profile or profile)

                            # No translation here — main.py returns the final language already.
                            print(f"[PIPE] Using TTS lang={msg_lang} | reply={reply_text}")

                            reply_for_tts = normalize_for_tts(reply_text or "", msg_lang)
                            try:
                                mp3 = await tts_elevenlabs(reply_for_tts, msg_lang)
                                if mp3:
                                    await send_whatsapp_audio(wa_from, mp3)
                                else:
                                    await send_whatsapp_text(wa_from, reply_text or "…")
                            except Exception as tts_err:
                                print("TTS error:", tts_err)
                                await send_whatsapp_text(wa_from, reply_text or "…")

                        except Exception as e:
                            print("Voice handling error:", e)
                            await send_whatsapp_text(
                                wa_from,
                                "There was an error processing your voice note. I'll reply in text for now."
                            )

    except Exception as e:
        print("Webhook processing error:", e)

    return JSONResponse({"status": "ok"})
