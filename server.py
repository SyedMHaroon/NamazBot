# server.py
import os
import json
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from dotenv import load_dotenv
load_dotenv()

from wa_client import send_whatsapp_text  # must be async
from session_store import get_profile, set_profile

# Import your LangGraph wrapper from main.py
from main import handle_turn


VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")

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


@app.post("/webhook")
async def whatsapp_inbound(request: Request):
    """
    Receive WhatsApp messages, route to LangGraph, and reply.
    """
    try:
        body: Dict[str, Any] = await request.json()

        # WhatsApp sends an array of "entry", each with "changes"
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # 1) Handle inbound messages
                for msg in value.get("messages", []):
                    wa_from = msg.get("from")  # sender phone number in wa format
                    text = extract_text(msg)

                    if not (wa_from and text):
                        continue

                    # Load + run one turn
                    profile = get_profile(wa_from)
                    reply_text, new_profile = await handle_turn(text, profile)

                    # Persist
                    set_profile(wa_from, new_profile)

                    # Reply
                    await send_whatsapp_text(wa_from, reply_text or "…")

                # 2) (Optional) You may inspect delivery/read statuses here
                # statuses = value.get("statuses", [])
                # for st in statuses: pass

    except Exception as e:
        # Log but still 200 so Meta doesn't keep retrying
        print("Webhook processing error:", e)

    return JSONResponse({"status": "ok"})


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
