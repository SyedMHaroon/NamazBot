# server.py
import os
import re
from typing import Any, Dict

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

# LangGraph turn handler
from main import handle_turn

VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")  # optional: EN→AR translate

WABASE = "https://graph.facebook.com/v20.0"

app = FastAPI()

# ---------- language helpers ----------
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

def detect_lang_from_text(s: str) -> str:
    """Return 'ar' if Arabic letters present, else 'en'."""
    if s and ARABIC_RE.search(s):
        return "ar"
    return "en"

def has_arabic(s: str) -> bool:
    return bool(s and ARABIC_RE.search(s))

def is_supported_lang(lang: str) -> bool:
    return (lang or "").lower() in ("en", "ar")

UNSUPPORTED_LANG_MSG = (
    "I currently support only English and Arabic. "
    "Please send your message in English or Arabic.\n\n"
    "أنا أدعم حاليًا اللغتين الإنجليزية والعربية فقط. "
    "من فضلك اكتب رسالتك باللغة الإنجليزية أو العربية."
)

async def translate_to_ar_gemini(text: str) -> str:
    """
    Translate English → Arabic using Gemini (if key present).
    Returns original text if key missing or on error.
    """
    if not GEMINI_API_KEY or not text:
        return text
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    prompt = (
        "Translate the following into clear Modern Standard Arabic. "
        "Keep numbers and proper nouns as-is. No commentary, no transliteration, no diacritics.\n\n"
        f"{text}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            out = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            return out or text
    except Exception as e:
        print("Gemini translate error:", e)
        return text

# ---------- health & verification ----------
@app.get("/")
async def health():
    return {"status": "ok"}

@app.get("/webhook")
async def whatsapp_verify(request: Request):
    """
    Meta verification endpoint. Echoes hub.challenge if VERIFY_TOKEN matches.
    """
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge)

    return PlainTextResponse("Verification failed", status_code=403)

# ---------- helpers ----------
def extract_text(msg: Dict[str, Any]) -> str:
    """
    Supports text and interactive messages (buttons/lists).
    """
    mtype = msg.get("type")

    if mtype == "text":
        return (msg.get("text", {}) or {}).get("body", "").strip()

    if mtype == "interactive":
        i = msg.get("interactive", {}) or {}
        itype = i.get("type")
        if itype == "button_reply":
            return (i.get("button_reply", {}) or {}).get("title", "").strip()
        if itype == "list_reply":
            return (i.get("list_reply", {}) or {}).get("title", "").strip()

    if "button" in msg:
        return (msg.get("button", {}) or {}).get("text", "").strip()

    return ""

async def fetch_whatsapp_media_bytes(media_id: str) -> bytes:
    """
    Step 1: GET /{media_id} to obtain a temporary media URL
    Step 2: GET the URL with Bearer token to download the bytes
    """
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

# ---------- webhook (messages) ----------
@app.post("/webhook")
async def whatsapp_inbound(request: Request):
    """
    Receive WhatsApp messages, route to LangGraph, and reply.
    Handles:
      - Text messages → text reply (auto Arabic/English based on message)
      - Voice notes (audio) → STT → LangGraph → TTS → audio reply (same language as STT)
    """
    try:
        body: Dict[str, Any] = await request.json()

        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {}) or {}

                for msg in value.get("messages", []) or []:
                    wa_from = msg.get("from")
                    if not wa_from:
                        continue

                    mtype = msg.get("type")

                    # ---- TEXT MESSAGE ----
                    if mtype == "text" or mtype == "interactive" or "button" in msg:
                        text = extract_text(msg)
                        if not text:
                            continue

                        # Detect language from incoming text
                        msg_lang = detect_lang_from_text(text)  # 'ar' or 'en'
                        if not is_supported_lang(msg_lang):
                            await send_whatsapp_text(wa_from, UNSUPPORTED_LANG_MSG)
                            continue

                        # Route one turn, hint language to the graph
                        profile = get_profile(wa_from) or {}
                        temp_profile = dict(profile)
                        temp_profile["lang"] = msg_lang

                        reply_text, new_profile = await handle_turn(text, temp_profile)
                        set_profile(wa_from, new_profile or profile)

                        # ALWAYS translate to Arabic if the user wrote in Arabic
                        if msg_lang == "ar":
                            reply_text = await translate_to_ar_gemini(reply_text or "")

                        # Log what we'll send
                        print(f"[BOT REPLY] lang={msg_lang} → {reply_text}")

                        # Normalize dates/prayer names and send
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
                            # 1) Download audio bytes
                            media_bytes = await fetch_whatsapp_media_bytes(media_id)

                            # 2) STT → text (pass MIME hint so transcode can guess container)
                            mime = (audio_obj.get("mime_type") or "").lower()
                            prefer_ext = ".ogg"
                            if "amr" in mime:
                                prefer_ext = ".amr"
                            elif "aac" in mime:
                                prefer_ext = ".aac"
                            elif "mp4" in mime or "m4a" in mime:
                                prefer_ext = ".m4a"
                            elif "mp3" in mime:
                                prefer_ext = ".mp3"

                            stt_text, lang_from_stt = stt_from_opus(
                                media_bytes, prefer_ext=prefer_ext
                            )
                            if not stt_text:
                                await send_whatsapp_text(
                                    wa_from,
                                    "I couldn't understand the voice message. Please try again."
                                )
                                continue

                            # Use STT-detected language for this turn
                            msg_lang = (lang_from_stt or "en").lower()
                            if not is_supported_lang(msg_lang):
                                await send_whatsapp_text(wa_from, UNSUPPORTED_LANG_MSG)
                                continue

                            # 3) Route turn via LangGraph (hint language)
                            profile = get_profile(wa_from) or {}
                            temp_profile = dict(profile)
                            temp_profile["lang"] = msg_lang

                            reply_text, new_profile = await handle_turn(stt_text, temp_profile)
                            set_profile(wa_from, new_profile or profile)

                            # ALWAYS translate to Arabic if the user spoke in Arabic
                            if msg_lang == "ar":
                                reply_text = await translate_to_ar_gemini(reply_text or "")

                            # Log what we'll speak
                            print(f"[PIPE] Using TTS lang={msg_lang} | reply={reply_text}")

                            # 4) TTS → audio (fallback to text if TTS fails)
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

                # Optional: inspect statuses
                # for st in (value.get("statuses") or []): pass

    except Exception as e:
        print("Webhook processing error:", e)

    return JSONResponse({"status": "ok"})
