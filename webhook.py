# file: webhook_server.py
import os
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "MY_SUPER_SECRET_TOKEN")

app = FastAPI()

@app.get("/webhook")
async def verify(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN and challenge:
        return PlainTextResponse(challenge, status_code=200)
    return PlainTextResponse("Forbidden", status_code=403)

# We'll add POST handling later once verified.
@app.post("/webhook")
async def receive_webhook(_request: Request):
    return PlainTextResponse("ok", status_code=200)

if __name__ == "__main__":
    # Run on 0.0.0.0 so ngrok can reach it
    uvicorn.run("webhook:app", host="0.0.0.0", port=8000, reload=True)
