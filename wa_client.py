# wa_client.py
import os
import httpx

WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

API_BASE = "https://graph.facebook.com/v20.0"


async def send_whatsapp_text(to: str, text: str) -> None:
    """
    Send a text message via WhatsApp Cloud API.
    """
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        print("⚠️ Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID; not sending WhatsApp reply.")
        return

    url = f"{API_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": (text or "")[:4096]},
    }

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print("WhatsApp send text error:", e.response.text)


async def upload_whatsapp_media(mp3_bytes: bytes, filename: str = "reply.mp3", mime: str = "audio/mpeg") -> str:
    """
    Uploads media to WhatsApp and returns media_id.
    """
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        raise RuntimeError("Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID.")

    url = f"{API_BASE}/{PHONE_NUMBER_ID}/media"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
    }
    # multipart/form-data: file + type + messaging_product
    files = {
        "file": (filename, mp3_bytes, mime),
    }
    data = {
        "messaging_product": "whatsapp",
        "type": mime,  # e.g., "audio/mpeg"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, data=data, files=files)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print("WhatsApp media upload error:", e.response.text)
            raise
        resp = r.json()
        media_id = resp.get("id")
        if not media_id:
            raise RuntimeError(f"Media upload returned no id: {resp}")
        return media_id


async def send_whatsapp_audio(to: str, mp3_bytes: bytes) -> None:
    """
    Upload audio to WhatsApp, then send it by media id.
    """
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        print("⚠️ Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID; not sending WhatsApp audio.")
        return

    try:
        media_id = await upload_whatsapp_media(mp3_bytes, filename="reply.mp3", mime="audio/mpeg")
    except Exception as e:
        print("Upload audio failed:", e)
        # Fallback to text if upload fails
        await send_whatsapp_text(to, "Audio reply failed to upload; sending text instead.")
        return

    url = f"{API_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "audio",
        "audio": {
            "id": media_id,
            # "ptt": True,  # uncomment if you want it to appear as a voice message bubble
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print("WhatsApp audio send error:", e.response.text)
            # Optional fallback:
            # await send_whatsapp_text(to, "Audio send failed; sending text instead.")
