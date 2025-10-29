# voice_pipeline.py
import os
import tempfile
import subprocess
import time
import re
from typing import Tuple, Optional

import httpx
from faster_whisper import WhisperModel

# ---------------- ENV CONFIG ----------------
ELEVEN_API_KEY     = os.getenv("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID_EN = os.getenv("ELEVEN_VOICE_ID_EN", "")  # set this in .env
ELEVEN_VOICE_ID_AR = os.getenv("ELEVEN_VOICE_ID_AR", "")  # set this in .env

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "small")   # small | medium
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")    # cpu | cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 is fast on CPU


# ---------------- TRANSCODE ----------------
def transcode_to_wav_bytes(input_bytes: bytes, prefer_ext: str = ".ogg") -> bytes:
    """Convert WhatsApp audio bytes to 16 kHz mono WAV."""
    candidates = [prefer_ext, ".ogg", ".amr", ".aac", ".m4a", ".mp3", ".mp4"]
    last_err = None

    for ext in candidates:
        inp_path = out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as inp, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                inp_path, out_path = inp.name, out.name
                inp.write(input_bytes)
                inp.flush()

            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", inp_path, "-ar", "16000", "-ac", "1", out_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if proc.returncode == 0:
                with open(out_path, "rb") as f:
                    return f.read()

            last_err = proc.stderr[-500:]  # tail for debug

        finally:
            for path in (inp_path, out_path):
                try:
                    if path:
                        os.unlink(path)
                except Exception:
                    pass

    raise RuntimeError(last_err or "ffmpeg transcode failed")


# ---------------- WHISPER (singleton) ----------------
_whisper_model: Optional[WhisperModel] = None

def get_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )
    return _whisper_model


def stt_from_opus(opus_bytes: bytes, prefer_ext: str = ".ogg") -> Tuple[str, str]:
    """
    Transcribe audio & detect language with faster-whisper.
    Returns (text, lang_code).
    """
    t0 = time.perf_counter()
    wav_bytes = transcode_to_wav_bytes(opus_bytes, prefer_ext=prefer_ext)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)

        model = get_whisper()
        segments, info = model.transcribe(
            wav_path,
            task="transcribe",
            vad_filter=True
        )
        text = " ".join(s.text.strip() for s in segments if s.text).strip()
        lang = (info.language or "en")

    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    print(f"[STT] lang={lang} text_len={len(text)} dur_ms={int((time.perf_counter() - t0) * 1000)}")
    return text, lang


# ---------------- TTS (ElevenLabs) ----------------
def _parse_quota_numbers(err_text: str) -> Tuple[int, int]:
    """
    Parse: "You have 39 credits remaining, while 1201 credits are required"
    Returns (remaining, required) or (-1, -1) if not found.
    """
    rem = req = -1
    m1 = re.search(r"You have\s+(\d+)\s+credits? remaining", err_text)
    m2 = re.search(r"while\s+(\d+)\s+credits? are required", err_text)
    if m1:
        try: rem = int(m1.group(1))
        except: pass
    if m2:
        try: req = int(m2.group(1))
        except: pass
    return rem, req


async def tts_elevenlabs(text: str, lang: str = "en") -> bytes:
    """
    Speak using separate voices for EN and AR. Only these two are supported.
    If ElevenLabs returns quota_exceeded, auto-trim to remaining credits and retry once.
    Return b'' on failure (caller should fall back to text).
    """
    if not ELEVEN_API_KEY or not text:
        return b""

    lang = (lang or "en").lower()
    if lang not in ("en", "ar"):
        print(f"[TTS] Unsupported language: {lang}")
        return b""

    voice_id = ELEVEN_VOICE_ID_EN if lang == "en" else ELEVEN_VOICE_ID_AR
    if not voice_id:
        print(f"[TTS] Missing voice ID for lang={lang}")
        return b""

    # Keep voice responses concise (WhatsApp UX + cost)
    if len(text) > 800:
        text = text[:800] + "…"

    model_id = "eleven_turbo_v2" if lang == "en" else "eleven_multilingual_v2"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }

    def _payload(t: str) -> dict:
        return {"text": t, "model_id": model_id}

    print(
        f"[TTS DBG] key_present={bool(ELEVEN_API_KEY)} "
        f"voice_id={(voice_id[:6] + '…') if voice_id else None} "
        f"model_id={model_id} len={len(text)} lang={lang}"
    )

    # Attempt 1
    try:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=_payload(text))
            r.raise_for_status()
            audio = r.content
        print(f"[TTS] lang={lang} chars={len(text)} dur_ms={int((time.perf_counter() - t0) * 1000)}")
        return audio

    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = e.response.text if e.response is not None else str(e)
        print("TTS HTTP error (attempt 1):", status, body)

        # Handle quota / payment-related cases
        if status in (401, 402, 429) and "quota_exceeded" in body:
            remaining, _required = _parse_quota_numbers(body)
            if remaining and remaining > 0:
                allowed = max(0, remaining - 5)  # small safety margin
                trimmed = (text[:allowed] + "…") if len(text) > allowed else text
                if allowed > 0:
                    print(f"[TTS] Retrying with trimmed text to {allowed} chars (remaining={remaining})")
                    try:
                        t0 = time.perf_counter()
                        async with httpx.AsyncClient(timeout=60) as client:
                            r2 = await client.post(url, headers=headers, json=_payload(trimmed))
                            r2.raise_for_status()
                            audio = r2.content
                        print(f"[TTS] lang={lang} chars={len(trimmed)} dur_ms={int((time.perf_counter() - t0) * 1000)}")
                        return audio
                    except Exception as e2:
                        print("TTS retry failed:", e2)
                        return b""
            return b""

        # Other HTTP errors → return empty
        return b""

    except Exception as e:
        print("TTS error:", e)
        return b""
