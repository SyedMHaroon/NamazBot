# wa_client.py
import os
import httpx

WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

async def send_whatsapp_text(to: str, text: str) -> None:
    """
    Send a text message via WhatsApp Cloud API.
    """
    if not (WHATSAPP_TOKEN and PHONE_NUMBER_ID):
        print("⚠️ Missing WHATSAPP_TOKEN or PHONE_NUMBER_ID; not sending WhatsApp reply.")
        return

    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
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
            print("WhatsApp send error:", e.response.text)
