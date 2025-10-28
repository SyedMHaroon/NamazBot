# server.py
import os
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

# LangGraph turn handler
from main import handle_turn

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

WABASE = "https://graph.facebook.com/v20.0"

app = FastAPI()


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


def extract_text(msg: Dict[str, Any]) -> str:
    """
    Supports text and interactive messages (buttons/lists).
    WhatsApp Cloud API message formats:
      - type == "text": msg["text"]["body"]
      - interactive button: msg["interactive"]["button_reply"]["title"]
      - interactive list:   msg["interactive"]["list_reply"]["title"]
      - older button type:  msg.get("button", {}).get("text")
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

    # Some older/alt payloads
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


@app.post("/webhook")
async def whatsapp_inbound(request: Request):
    """
    Receive WhatsApp messages, route to LangGraph, and reply.
    Handles:
      - Text messages → text reply
      - Voice notes (audio) → STT → LangGraph → TTS → audio reply (fallback to text)
    """
    try:
        body: Dict[str, Any] = await request.json()

        # WhatsApp sends an array of "entry", each with "changes"
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {}) or {}

                # 1) Handle inbound messages
                for msg in value.get("messages", []) or []:
                    wa_from = msg.get("from")  # sender phone in WA format
                    if not wa_from:
                        continue

                    mtype = msg.get("type")

                    # ---- TEXT MESSAGE ----
                    if mtype == "text" or mtype == "interactive" or "button" in msg:
                        text = extract_text(msg)
                        if not text:
                            continue

                        # Run one turn
                        profile = get_profile(wa_from)
                        reply_text, new_profile = await handle_turn(text, profile)

                        # Persist + reply
                        set_profile(wa_from, new_profile)
                        await send_whatsapp_text(wa_from, reply_text or "…")
                        continue

                    # ---- VOICE NOTE / AUDIO ----
                    if mtype == "audio":
                        audio_obj = (msg.get("audio") or {})
                        media_id = audio_obj.get("id")
                        if not media_id:
                            continue

                        try:
                            # 1) Download audio bytes (e.g., OGG/OPUS/AMR/AAC/M4A)
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

                            stt_text, lang = stt_from_opus(media_bytes, prefer_ext=prefer_ext)
                            if not stt_text:
                                await send_whatsapp_text(
                                    wa_from,
                                    "I couldn't understand the voice message. Please try again."
                                )
                                continue

                            # 3) Route turn via LangGraph
                            profile = get_profile(wa_from)
                            reply_text, new_profile = await handle_turn(stt_text, profile)
                            set_profile(wa_from, new_profile)

                            # 4) TTS → audio (fallback to text if TTS fails)
                            try:
                                mp3 = await tts_elevenlabs(reply_text)
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

                # 2) Optional: inspect delivery/read statuses
                # for st in (value.get("statuses") or []):
                #     pass

    except Exception as e:
        # Log but still 200 so Meta doesn't keep retrying
        print("Webhook processing error:", e)

    return JSONResponse({"status": "ok"})
