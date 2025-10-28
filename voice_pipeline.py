# voice_pipeline.py
import os, tempfile, subprocess, time
from typing import Tuple, Optional

import httpx
from faster_whisper import WhisperModel

ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID  = os.getenv("ELEVEN_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "small")   # small | medium
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")    # cpu | cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")  # int8 is fast on CPU

# ---------------- Transcode (resilient) ----------------
def transcode_to_wav_bytes(input_bytes: bytes, prefer_ext: str = ".ogg") -> bytes:
    """
    Transcodes WhatsApp audio bytes to 16kHz mono WAV.
    prefer_ext hints the container ('.ogg', '.amr', '.aac', '.m4a', '.mp3', '.mp4').
    Tries a few containers and surfaces ffmpeg stderr on failure.
    """
    candidates = [prefer_ext, ".ogg", ".amr", ".aac", ".m4a", ".mp3", ".mp4"]
    last_err = None

    for ext in candidates:
        inp_path = out_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as inp, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                inp_path, out_path = inp.name, out.name
                inp.write(input_bytes); inp.flush()
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", inp_path, "-ar", "16000", "-ac", "1", out_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if proc.returncode == 0:
                with open(out_path, "rb") as f:
                    return f.read()
            last_err = f"[{ext}] ffmpeg stderr (tail):\n{proc.stderr[-1200:]}"
        finally:
            try:
                if inp_path: os.unlink(inp_path)
            except Exception:
                pass
            try:
                if out_path: os.unlink(out_path)
            except Exception:
                pass

    raise RuntimeError(last_err or "ffmpeg transcode failed")

# ---------------- Whisper model (singleton) ----------------
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

# ---------------- STT ----------------

def stt_from_opus(opus_bytes: bytes, prefer_ext: str = ".ogg") -> Tuple[str, str]:
    """
    Windows-safe: write WAV to a temp path with delete=False, close it,
    transcribe, then unlink.
    """
    t0 = time.perf_counter()

    # robust transcode (you already have this helper)
    wav_bytes = transcode_to_wav_bytes(opus_bytes, prefer_ext=prefer_ext)

    # create a temp WAV path, CLOSE the handle before whisper opens it
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)  # file is closed as we exit this with-block

        model = get_whisper()
        # Now the file is closed; Whisper can open it safely on Windows
        segments, info = model.transcribe(
            wav_path,
            task="transcribe",
            vad_filter=True
        )

        text = " ".join(s.text.strip() for s in segments if s.text).strip()
        lang = (info.language or "ar")

    finally:
        # Always clean up the temp file
        try:
            os.unlink(wav_path)
        except Exception:
            pass

    t1 = time.perf_counter()
    print(f"[STT] lang={lang} text_len={len(text)} dur_ms={int((t1 - t0)*1000)}")
    return text, lang

# ---------------- TTS (ElevenLabs) ----------------
async def tts_elevenlabs(text: str, retries: int = 1) -> bytes:
    """
    Return MP3 bytes for given text via ElevenLabs.
    Returns b'' if no API key or if request fails.
    """
    if not ELEVEN_API_KEY:
        return b""
    if not text:
        return b""

    # Avoid super long messages in POC (WhatsApp voice UX)
    if len(text) > 1200:
        text = text[:1200] + "…"

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {"text": text, "model_id": "eleven_multilingual_v2"}

    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                audio = r.content
            t1 = time.perf_counter()
            print(f"[TTS] chars={len(text)} dur_ms={int((t1 - t0)*1000)}")
            return audio
        except httpx.HTTPStatusError as e:
            # 429/5xx transient? try once more
            if attempt < retries and (e.response.status_code >= 500 or e.response.status_code == 429):
                await httpx.AsyncClient().aclose()  # no-op; keep structure consistent
                continue
            print("TTS HTTP error:", getattr(e.response, "text", str(e)))
            break
        except Exception as e:
            if attempt < retries:
                continue
            print("TTS error:", e)
            break
    return b""
