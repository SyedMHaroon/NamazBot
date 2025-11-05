"""
Simple LangGraph bot — onboarding + context-aware (no server/web code here).
- Collects: Name, Email, Location ("City - Country")
- Routes via LLM classifier (JSON): islamic_date | prayer_times | next_prayer | general
- Validates location against Aladhan (shared HTTP client + small cache)
- Uses Gemini (GEMINI_API_KEY) for classification + general answers
- Language-aware replies: Arabic if profile["lang"] == "ar", else English
- Auto-detects Arabic per turn if profile["lang"] not explicitly set
- Accepts optional `context` from server: short_history + semantic_snippets
"""

import os
import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, Literal, TypedDict, Optional, Any, List, Tuple
from zoneinfo import ZoneInfo
from time import time

import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import pycountry

load_dotenv()

# -------------------------
# Gemini setup
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
)

# Digest schedule and deduplication settings
DIGEST_HOUR = int(os.getenv("DIGEST_HOUR"))
DIGEST_MINUTE = int(os.getenv("DIGEST_MINUTE"))
DIGEST_DEDUPE = os.getenv("DIGEST_DEDUPE")
# -------------------------
# Shared HTTP client + tiny cache for Aladhan
# -------------------------
HTTP = httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(15.0, connect=5.0))

# cache key: (city or "", country or "", date or "")
_ALADHAN_CACHE: dict[tuple[str, str, str], tuple[float, dict]] = {}
_ALADHAN_TTL = int(os.getenv("ALADHAN_TTL_SECONDS", "600"))  # 10 min default

def _cache_get(key: tuple[str, str, str]) -> Optional[dict]:
    item = _ALADHAN_CACHE.get(key)
    if not item:
        return None
    ts, data = item
    if time() - ts > _ALADHAN_TTL:
        _ALADHAN_CACHE.pop(key, None)
        return None
    return data

def _cache_set(key: tuple[str, str, str], data: dict) -> None:
    _ALADHAN_CACHE[key] = (time(), data)

INTENT_LABELS = ["islamic_date", "prayer_times", "next_prayer", "reminder", "calendar_connect", "calendar_create_event", "calendar_view_events", "calendar_find_events", "calendar_delete_event", "general"]
PRAYER_NAMES  = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]
PRAYER_ORDER  = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]

# -------------------------
# State definition
# -------------------------
class BotState(TypedDict, total=False):
    question: str
    intent: Literal["islamic_date", "prayer_times", "next_prayer", "reminder", "calendar_connect", "calendar_create_event", "calendar_view_events", "calendar_find_events", "calendar_delete_event", "general"]
    profile: Dict[str, str]     # name, email, city, country, tz, lang, plus temp flags
    context: Dict[str, Any]     # {"short_history": [(role,text),...], "semantic_snippets": [str,...]}
    reply: str
    wa_id: Optional[str]         # WhatsApp user ID, needed for reminders

# -------------------------
# Helpers: API + parsing
# -------------------------
RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

async def aladhan_fetch(city: Optional[str], country: Optional[str], date: Optional[str] = None) -> Dict[str, Any]:
    """
    If both city and country are present -> use timingsByCity.
    If only city is present -> use timingsByAddress (no guessing a country).
    Cached for a short TTL to reduce latency/cost.
    """
    if not city:
        raise ValueError("aladhan_fetch requires at least a city.")

    base_params = {"method": 2}
    if date:
        base_params["date"] = date

    key = (city or "", country or "", date or "")
    cached = _cache_get(key)
    if cached:
        return cached

    if city and country:
        url = "https://api.aladhan.com/v1/timingsByCity"
        params = {**base_params, "city": city, "country": country}
    else:
        url = "https://api.aladhan.com/v1/timingsByAddress"
        params = {**base_params, "address": city}

    r = await HTTP.get(url, params=params)
    r.raise_for_status()
    data = r.json()["data"]
    _cache_set(key, data)
    return data

async def aladhan(city: str, country: str, date: Optional[str] = None) -> Dict[str, Any]:
    key = (city or "", country or "", date or "")
    cached = _cache_get(key)
    if cached:
        return cached

    url = "https://api.aladhan.com/v1/timingsByCity"
    params = {"city": city, "country": country, "method": 2}
    if date:
        params["date"] = date
    r = await HTTP.get(url, params=params)
    r.raise_for_status()
    data = r.json()["data"]
    _cache_set(key, data)
    return data

def clean_time(t: str) -> str:
    m = re.search(r"(\d{1,2}:\d{2})", t)
    return m.group(1) if m else t

def normalize_country_name(name: str) -> Optional[str]:
    """Return official country name if recognized; else None."""
    if not name:
        return None
    n = name.strip()
    aliases = {
        "KSA": "Saudi Arabia", "UAE": "United Arab Emirates",
        "UK": "United Kingdom", "USA": "United States",
        "US": "United States", "PK": "Pakistan",
        "Ksa": "Saudi Arabia", "Uae": "United Arab Emirates",
        "Uk": "United Kingdom", "Usa": "United States", "Pk": "Pakistan",
    }
    if n in aliases:
        return aliases[n]
    try:
        for c in pycountry.countries:
            if n.lower() == c.name.lower():
                return c.name
            if hasattr(c, "common_name") and n.lower() == c.common_name.lower():
                return c.common_name
        for c in pycountry.countries:
            if c.name.lower().startswith(n.lower()):
                return c.name
    except Exception:
        pass
    return None

def parse_city_country(line: str) -> tuple[Optional[str], Optional[str]]:
    """Expected 'City - Country' → (City, CountryName) title-cased; returns (None, None) if malformed."""
    if "-" not in line:
        return None, None
    left, right = line.split("-", 1)
    city_raw = left.strip()
    country_raw = right.strip()
    if len(city_raw) < 2 or len(country_raw) < 2:
        return None, None
    city = city_raw.title()
    country_norm = normalize_country_name(country_raw)
    if not country_norm:
        return city, None
    return city, country_norm

def find_country_in_text(text: str) -> Optional[str]:
    """Return normalized country name ONLY if the user explicitly typed it."""
    if not text:
        return None
    t = text.lower()
    alias_map = {
        "ksa": "Saudi Arabia", "uae": "United Arab Emirates",
        "uk": "United Kingdom", "usa": "United States",
        "us": "United States", "pk": "Pakistan",
    }
    for alias, fullname in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", t):
            return fullname
    try:
        for c in pycountry.countries:
            name = c.name.lower()
            if re.search(rf"\b{re.escape(name)}\b", t):
                return c.name
            if hasattr(c, "common_name"):
                cname = c.common_name.lower()
                if re.search(rf"\b{re.escape(cname)}\b", t):
                    return c.common_name
    except Exception:
        pass
    return None

def get_effective_location(state: BotState) -> tuple[Optional[str], Optional[str], bool]:
    """
    Returns (city, country, address_mode).
    - address_mode=True => query Aladhan by address (city only); do NOT append a country in the reply.
    """
    prof = state.get("profile", {}) or {}
    o_city = prof.get("_override_city")
    o_country = prof.get("_override_country")
    p_city = prof.get("city")
    p_country = prof.get("country")

    if o_city and not o_country:
        return o_city, None, True

    city = o_city or p_city
    country = o_country or p_country
    return city, country, False

def clear_overrides(state: BotState) -> None:
    prof = state.get("profile", {}) or {}
    prof.pop("_override_city", None)
    prof.pop("_override_country", None)
    state["profile"] = prof

async def validate_location_against_api(city: str, country: str) -> tuple[bool, Optional[str]]:
    """Probe Aladhan. Return (ok, timezone)."""
    try:
        data = await aladhan(city, country)
        t = (data or {}).get("timings") or {}
        tz = (data or {}).get("meta", {}).get("timezone")
        required = {"Fajr","Dhuhr","Asr","Maghrib","Isha"}
        ok = bool(t) and required.issubset(set(t.keys())) and bool(tz)
        return ok, tz
    except Exception:
        return False, None

def _safe_json_extract(text: str) -> dict:
    """Best-effort extraction of a single JSON object from LLM text."""
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                return {}
        return {}

# -------------------------
# APScheduler setup
# -------------------------
_scheduler: Optional[AsyncIOScheduler] = None

def get_scheduler() -> AsyncIOScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler

async def start_scheduler():
    """Start the scheduler if not already running."""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
        print("[SCHED] APScheduler started")

async def stop_scheduler():
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        _scheduler = None
        print("[SCHED] APScheduler stopped")

def setup_digest_scheduler(get_profile_fn, send_text_fn, hour: int = DIGEST_HOUR, minute: int = DIGEST_MINUTE, dedupe: bool = DIGEST_DEDUPE):
    """
    Schedule daily digest at hour:minute (default 16:09) in each user's timezone.
    This runs digest_job.run_digest_tick for all subscribed users.
    
    Args:
        dedupe: If True, skip if already sent today. If False, always send (for testing).
    """
    from digest_job import run_digest_tick
    scheduler = get_scheduler()
    
    # Schedule to run every minute - run_digest_tick will check if it's the right time for each user
    scheduler.add_job(
        run_digest_tick,
        trigger=CronTrigger(minute="*"),  # Every minute
        args=[get_profile_fn, send_text_fn, hour, minute, dedupe],
        id="daily_digest",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,  # Execute if missed within 60 seconds
        coalesce=True  # Combine multiple missed runs into one
    )
    print(f"[SCHED] Digest scheduled to run at {hour:02d}:{minute:02d} in each user's timezone (dedupe={dedupe})")

def setup_reminder_scheduler(send_text_fn):
    """
    Schedule reminder tick to run every minute.
    This processes reminders from Redis.
    """
    from digest_job import run_reminder_tick
    scheduler = get_scheduler()
    
    scheduler.add_job(
        run_reminder_tick,
        trigger=CronTrigger(minute="*"),  # Every minute
        args=[send_text_fn],
        id="reminder_tick",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,  # Execute if missed within 60 seconds
        coalesce=True  # Combine multiple missed runs into one
    )
    print("[SCHED] Reminder tick scheduled to run every minute")

def setup_prayer_reminder_scheduler(get_profile_fn, send_text_fn):
    """
    Schedule prayer reminder tick to run every minute.
    This checks for upcoming prayers and sends reminders 10 minutes before each prayer.
    """
    from digest_job import run_prayer_reminder_tick
    scheduler = get_scheduler()
    
    scheduler.add_job(
        run_prayer_reminder_tick,
        trigger=CronTrigger(minute="*"),  # Every minute
        args=[get_profile_fn, send_text_fn],
        id="prayer_reminder_tick",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,  # Execute if missed within 60 seconds
        coalesce=True  # Combine multiple missed runs into one
    )
    print("[SCHED] Prayer reminder tick scheduled to run every minute (10 minutes before each prayer)")

# -------------------------
# Language helpers
# -------------------------
AR_RE = re.compile(r"[\u0600-\u06FF]")

def _lang(state: BotState) -> str:
    prof = state.get("profile", {}) or {}
    return (prof.get("lang") or "en").lower()

def _has_ar(text: str) -> bool:
    return bool(text and AR_RE.search(text))

async def _ensure_output_language(state: BotState, text: str) -> str:
    """
    If profile.lang == 'ar', translate final surface text to Arabic (MSA) using Gemini.
    Otherwise return as-is.
    """
    lang = _lang(state)
    if lang != "ar":
        return text or ""

    if not text:
        return ""

    if _has_ar(text):
        return text

    prompt = (
        "Translate the following into clear Modern Standard Arabic. "
        "Keep numbers and proper nouns as-is. No commentary, no transliteration, no diacritics.\n\n"
        f"{text}"
    )
    try:
        res = await llm.ainvoke(prompt)
        out = (res.content or "").strip()
        return out or text
    except Exception:
        return text

# Auto-detect language per turn (from user input)
def _auto_set_lang(profile: Dict[str, str], question: str) -> Dict[str, str]:
    prof = dict(profile or {})
    prof["lang"] = "ar" if _has_ar(question) else "en"
    return prof

# -------------------------
# Intent classification (LLM router)
# -------------------------
async def llm_intent_json(question: str, context: Optional[Dict[str, Any]] = None) -> dict:
    """
    Ask Gemini for strict JSON: {"intent": "...", "slots": {"prayer_name":..., "date":..., "city":..., "country":...}}
    IMPORTANT: The model must NOT infer country; only set it if the user explicitly typed it.
    We allow the last few user turns as faint signal but keep router simple.
    """
    # Light-touch use of history: include last 2 user messages (if provided)
    history_lines: List[str] = []
    if context:
        sh = context.get("short_history") or []
        # sh is expected as [(role,text), ...]
        for role, txt in sh[-4:]:  # last 4 entries max to keep router cheap
            if role == "user" and txt:
                history_lines.append(f"- {txt}")
    history_block = "\n".join(history_lines)

    system = (
        "You are a router that ONLY returns strict JSON. No prose, no markdown.\n"
        "Allowed intents: islamic_date, prayer_times, next_prayer, reminder, calendar_connect, calendar_create_event, calendar_view_events, calendar_find_events, calendar_delete_event, general.\n"
        "IMPORTANT: ANY question about dates, today's date, Hijri date, Islamic date, Gregorian date, or 'what date' should ALWAYS be 'islamic_date'.\n"
        "Calendar intents:\n"
        "  - calendar_connect: When user wants to connect/link their Google Calendar (e.g., 'connect calendar', 'link calendar', 'setup calendar')\n"
        "  - calendar_create_event: When user wants to create/add an event (e.g., 'create event', 'add meeting', 'schedule appointment')\n"
        "  - calendar_view_events: When user wants to see upcoming events (e.g., 'show events', 'view calendar', 'my events')\n"
        "  - calendar_find_events: When user wants to search for specific events (e.g., 'find events', 'search calendar')\n"
        "  - calendar_delete_event: When user wants to delete/cancel an event (e.g., 'delete event', 'cancel meeting')\n"
        "Slots (all optional):\n"
        "  - prayer_name: Fajr|Dhuhr|Asr|Maghrib|Isha\n"
        "  - date: today|tomorrow|YYYY-MM-DD\n"
        "  - city: string (ONLY if user typed a city)\n"
        "  - country: string (ONLY if user explicitly typed a country; DO NOT GUESS OR INFER)\n"
        "  - reminder_text: string (ONLY for reminder intent; the text to remind about)\n"
        "  - reminder_time: string (ONLY for reminder intent; time like 'HH:MM' or relative like 'in 30 minutes')\n"
        "Use 'reminder' intent when user asks to set a reminder, alarm, or scheduled notification.\n"
        "Use 'islamic_date' for ANY date-related query (today, what date, Hijri, Gregorian, etc.).\n"
        "Use 'general' ONLY for questions that don't fit any other category - and note that the bot will politely decline general chat.\n"
        "If unsure about ANY slot, set it to null. DO NOT invent or infer countries.\n"
        "Respond with exactly this JSON schema:\n"
        "{\n"
        '  "intent": "islamic_date|prayer_times|next_prayer|reminder|calendar_connect|calendar_create_event|calendar_view_events|calendar_find_events|calendar_delete_event|general",\n'
        '  "slots": {"prayer_name": null|string, "date": null|string, "city": null|string, "country": null|string, "reminder_text": null|string, "reminder_time": null|string}\n'
        "}\n"
    )
    prompt = f"{system}\nRecent user messages:\n{history_block}\n\nCurrent user: {question}\n"
    res = await llm.ainvoke(prompt)
    data = _safe_json_extract((res.content or "").strip())

    intent = str(data.get("intent", "general")).lower()
    if intent not in INTENT_LABELS:
        intent = "general"

    slots = data.get("slots", {}) or {}
    pn = slots.get("prayer_name")
    if pn:
        mapping = {"fajr":"Fajr","zuhr":"Dhuhr","dhuhr":"Dhuhr","asr":"Asr","maghrib":"Maghrib","isha":"Isha"}
        slots["prayer_name"] = mapping.get(str(pn).strip().lower(), None)
        if slots["prayer_name"] not in PRAYER_NAMES:
            slots["prayer_name"] = None
    for k in ("city","country"):
        if isinstance(slots.get(k), str):
            val = slots[k].strip()
            slots[k] = val.title()
    return {"intent": intent, "slots": slots}

# -------------------------
# Onboarding / Profile collection
# -------------------------
REQUIRED_FIELDS = ["name", "email", "city", "country"]

def _profile_complete(prof: dict) -> bool:
    return all(prof.get(k) for k in REQUIRED_FIELDS)

async def ensure_profile(state: BotState) -> BotState:
    """
    Sequentially ask: Name → Email → Location ("City - Country").
    Handles confirmation cleanly so once the user says 'yes', setup finishes.
    """
    profile = state.get("profile", {}) or {}
    q = (state.get("question", "") or "").strip()

    # If already complete, let downstream nodes answer.
    if profile.get("_setup_done") and _profile_complete(profile):
        state["profile"] = profile
        return state

    # --- Location confirmation step ---
    if profile.get("_confirm_loc"):
        if q.lower() in {"yes", "y", "haan", "han", "ji", "ok"}:
            staged = profile.get("_staged_loc", {})
            profile["city"] = staged.get("city")
            profile["country"] = staged.get("country")
            if staged.get("tz"):
                profile["tz"] = staged.get("tz")

            profile.pop("_staged_loc", None)
            profile.pop("_confirm_loc", None)
            profile.pop("_await", None)
            profile["_setup_done"] = True
            
            # Automatically subscribe user to daily digest
            wa_id = state.get("wa_id")
            if wa_id:
                try:
                    from digest_job import subscribe_to_digest
                    await subscribe_to_digest(wa_id)
                except Exception as e:
                    print(f"[SCHED] Failed to auto-subscribe {wa_id} to digest: {e}")
            
            state["reply"] = (
                f"Shukriya, {profile.get('name', '')}! Setup complete for "
                f"{profile['city']}, {profile['country']}.\n"
                "You can now ask: 'Fajr time', 'Next prayer', or 'Islamic date'.\n"
                "You'll also receive daily prayer time digests."
            )
            state["profile"] = profile
            return state

        elif q.lower() in {"no", "n", "na", "nah"}:
            profile.pop("_staged_loc", None)
            profile.pop("_confirm_loc", None)
            profile["_await"] = "location"
            state["reply"] = "Please enter your location as: City - Country  (e.g., Lahore - Pakistan)"
            state["profile"] = profile
            return state

        else:
            state["reply"] = "Please reply Yes or No to confirm your location."
            state["profile"] = profile
            return state

    # --- Field collection flow ---
    awaiting = profile.get("_await")

    if awaiting == "name":
        if not q or q.lower() in {"ok", "okay"}:
            state["reply"] = "Please enter your Name:"
            state["profile"] = profile
            return state
        profile["name"] = q
        profile.pop("_await", None)

    elif awaiting == "email":
        if not RE_EMAIL.match(q):
            state["reply"] = "Please enter a valid email (e.g., name@example.com)."
            state["profile"] = profile
            return state
        profile["email"] = q
        profile.pop("_await", None)

    elif awaiting == "location":
        city, country = parse_city_country(q)
        if not city or not country:
            state["reply"] = "Use the format: City - Country  (e.g., Lahore - Pakistan)"
            state["profile"] = profile
            return state

        ok, tz = await validate_location_against_api(city, country)
        if not ok:
            state["reply"] = (
                "I couldn't validate that location. "
                "Please re-enter as: City - Country  (e.g., Karachi - Pakistan)"
            )
            profile["_await"] = "location"
            state["profile"] = profile
            return state

        profile["_staged_loc"] = {"city": city, "country": country, "tz": tz}
        profile["_confirm_loc"] = True
        profile.pop("_await", None)
        state["reply"] = (
            f"Please confirm your location: {city}, {country} (timezone: {tz}).\n"
            "Reply **Yes** to confirm or **No** to change."
        )
        state["profile"] = profile
        return state

    # --- Ask next missing field ---
    if not profile.get("name"):
        profile["_await"] = "name"
        state["reply"] = "Please enter your Name:"
        state["profile"] = profile
        return state

    if not profile.get("email"):
        profile["_await"] = "email"
        state["reply"] = "Please enter your Email:"
        state["profile"] = profile
        return state

    if not (profile.get("city") and profile.get("country")):
        profile["_await"] = "location"
        state["reply"] = "Please enter your location as: City - Country  (e.g., Lahore - Pakistan)"
        state["profile"] = profile
        return state

    # --- Final confirmation once all info present ---
    if not profile.get("_setup_done"):
        profile["_setup_done"] = True
        # Automatically subscribe user to daily digest
        wa_id = state.get("wa_id")
        if wa_id:
            try:
                from digest_job import subscribe_to_digest
                await subscribe_to_digest(wa_id)
            except Exception as e:
                print(f"[SCHED] Failed to auto-subscribe {wa_id} to digest: {e}")
        
        state["reply"] = (
            f"Shukriya, {profile['name']}! Setup complete for "
            f"{profile['city']}, {profile['country']}.\n"
            "You can now ask: 'Fajr time', 'Next prayer', or 'Islamic date'.\n"
            "You'll also receive daily prayer time digests."
        )

    # Preserve wa_id through state updates
    existing_wa_id = state.get("wa_id")
    state["profile"] = profile
    if existing_wa_id:
        state["wa_id"] = existing_wa_id
    return state

# -------------------------
# Router
# -------------------------
async def classify(state: BotState) -> BotState:
    prof = state.get("profile", {}) or {}
    if prof.get("_await") or not _profile_complete(prof) or prof.get("_onboarding_ack") or prof.get("_confirm_loc"):
        state["intent"] = "general"
        return state

    q = state.get("question", "") or ""
    
    # Pre-route date queries directly to islamic_date (before LLM classification)
    q_lower = q.lower()
    date_keywords = [
        "date", "today", "what day", "hijri", "islamic date", "gregorian",
        "التاريخ", "اليوم", "هجري", "ميلادي", "what's the date", "what is the date",
        "current date", "today date", "today's date"
    ]
    if any(keyword in q_lower for keyword in date_keywords):
        state["intent"] = "islamic_date"
        state["profile"] = prof
        return state
    
    data = await llm_intent_json(q, context=state.get("context"))
    label = data["intent"]
    slots = data.get("slots", {}) or {}

    if slots.get("city"):
        prof["_override_city"] = slots["city"]
    else:
        prof.pop("_override_city", None)

    explicit_country = find_country_in_text(q)
    if explicit_country:
        prof["_override_country"] = normalize_country_name(explicit_country) or explicit_country
    else:
        prof.pop("_override_country", None)

    if slots.get("prayer_name"):
        prof["_requested_prayer"] = slots["prayer_name"]
    else:
        prof.pop("_requested_prayer", None)

    if slots.get("date"):
        prof["_requested_date"] = slots["date"]
    else:
        prof.pop("_requested_date", None)

    if slots.get("reminder_text"):
        prof["_reminder_text"] = slots["reminder_text"]
    else:
        prof.pop("_reminder_text", None)

    if slots.get("reminder_time"):
        prof["_reminder_time"] = slots["reminder_time"]
    else:
        prof.pop("_reminder_time", None)

    # Preserve wa_id through state updates
    existing_wa_id = state.get("wa_id")
    state["profile"] = prof
    state["intent"] = label
    if existing_wa_id:
        state["wa_id"] = existing_wa_id
    return state

# -------------------------
# UI: visualization
# -------------------------
def mermaid_diagram() -> str:
    return """flowchart TD
    start([entry]) --> ensure_profile
    ensure_profile -->|missing fields| noop([ask next field & END])
    ensure_profile -->|complete| classify
    classify -->|islamic_date| islamic_date
    classify -->|prayer_times| prayer_times
    classify -->|next_prayer| next_prayer
    classify -->|reminder| scheduler_agent
    classify -->|calendar_connect| calendar_connect
    classify -->|calendar_create_event| calendar_create_event
    classify -->|calendar_view_events| calendar_view_events
    classify -->|calendar_find_events| calendar_find_events
    classify -->|calendar_delete_event| calendar_delete_event
    classify -->|general| general
    islamic_date --> END((END))
    prayer_times --> END
    next_prayer --> END
    scheduler_agent --> END
    calendar_connect --> END
    calendar_create_event --> END
    calendar_view_events --> END
    calendar_find_events --> END
    calendar_delete_event --> END
    general --> END
    """

# -------------------------
# Task nodes (language-aware)
# -------------------------
async def islamic_date(state: BotState) -> BotState:
    city, country, address_mode = get_effective_location(state)
    d = await aladhan_fetch(city, country, None)
    hijri = d["date"]["hijri"]["date"]
    greg  = d["date"]["readable"]

    place = city if (address_mode or not country) else f"{city}, {country}"
    base = f"Islamic (Hijri) date in {place}: {hijri}\nGregorian: {greg}"
    state["reply"] = await _ensure_output_language(state, base)

    clear_overrides(state)
    return state

async def prayer_times(state: BotState) -> BotState:
    city, country, address_mode = get_effective_location(state)

    date_req = state["profile"].get("_requested_date")
    date_param: Optional[str] = None
    if date_req:
        s = date_req.strip().lower()
        if s == "today":
            date_param = None
        elif s == "tomorrow":
            d0 = await aladhan_fetch(city, country, None)
            tzname = d0.get("meta", {}).get("timezone", "UTC")
            now = datetime.now(ZoneInfo(tzname))
            date_param = (now + timedelta(days=1)).strftime("%d-%m-%Y")
        else:
            try:
                dt = datetime.strptime(date_req, "%Y-%m-%d")
                date_param = dt.strftime("%d-%m-%Y")
            except Exception:
                date_param = None

    d = await aladhan_fetch(city, country, date_param)
    t = {k: clean_time(v) for k, v in d["timings"].items()}
    req = state["profile"].get("_requested_prayer")

    place = city if (address_mode or not country) else f"{city}, {country}"

    if req in PRAYER_NAMES:
        base = f"{req} time in {place}: {t.get(req, 'N/A')}"
        state["reply"] = await _ensure_output_language(state, base)
        clear_overrides(state)
        return state

    lines = [f"{k}: {t.get(k, 'N/A')}" for k in PRAYER_ORDER]
    when = "today" if not date_param else (state["profile"].get("_requested_date") or "the selected date")
    base = f"Prayer times {when} for {place}:\n" + "\n".join(lines)
    state["reply"] = await _ensure_output_language(state, base)

    clear_overrides(state)
    return state

async def next_prayer(state: BotState) -> BotState:
    city, country, address_mode = get_effective_location(state)

    d = await aladhan_fetch(city, country, None)
    tzname = (state.get("profile", {}) or {}).get("tz") or d.get("meta", {}).get("timezone", "UTC")
    tz     = ZoneInfo(tzname)
    now    = datetime.now(tz)
    t = {k: clean_time(v) for k, v in d["timings"].items()}

    def to_dt(hhmm: str) -> datetime:
        h, m = map(int, hhmm.split(":"))
        return datetime(now.year, now.month, now.day, h, m, tzinfo=tz)

    nxt_name, nxt_time = None, None
    for p in PRAYER_ORDER:
        if to_dt(t[p]) > now:
            nxt_name, nxt_time = p, to_dt(t[p])
            break

    if not nxt_name:
        # after Isha → tomorrow's Fajr
        tomorrow = (now + timedelta(days=1)).strftime("%d-%m-%Y")
        d2 = await aladhan_fetch(city, country, tomorrow)
        fajr = clean_time(d2["timings"]["Fajr"])
        h, m = map(int, fajr.split(":"))
        nxt_name = "Fajr"
        nxt_time = datetime(now.year, now.month, now.day, h, m, tzinfo=tz) + timedelta(days=1)

    rem = nxt_time - now
    minutes_total = int(rem.total_seconds() // 60)
    hours, rem_mins = divmod(minutes_total, 60)

    place = city if (address_mode or not country) else f"{city}, {country}"
    base = f"Next prayer in {place}: {nxt_name} at {nxt_time.strftime('%H:%M')} ({hours}h {rem_mins}m left)"
    state["reply"] = await _ensure_output_language(state, base)

    clear_overrides(state)
    return state

async def scheduler_agent(state: BotState) -> BotState:
    """
    Handle reminder creation requests.
    Parses reminder_text and reminder_time from profile, enqueues reminder to Redis.
    """
    from digest_job import enqueue_reminder
    
    q = state.get("question", "")
    prof = state.get("profile", {}) or {}
    lang = _lang(state)
    
    # Get reminder slots from profile (set by classify node)
    reminder_text = prof.get("_reminder_text") or ""
    reminder_time_str = prof.get("_reminder_time") or ""
    
    # If not extracted from slots, try to parse from question using LLM
    if not reminder_text or not reminder_time_str:
        # Use LLM to extract reminder details
        prompt = (
            f"Extract reminder details from: {q}\n"
            "Return JSON: {\"text\": \"reminder text\", \"time\": \"HH:MM\" or \"in X minutes\" or \"in X hours\"}\n"
            "If time is relative (e.g., 'in 30 minutes'), parse it. If absolute (e.g., '14:30'), use it.\n"
            "If unclear, return {\"text\": null, \"time\": null}"
        )
        try:
            res = await llm.ainvoke(prompt)
            data = _safe_json_extract((res.content or "").strip())
            reminder_text = reminder_text or data.get("text", "")
            reminder_time_str = reminder_time_str or data.get("time", "")
        except Exception as e:
            print(f"[SCHED] LLM reminder extraction failed: {e}")
    
    # Validate reminder data
    if not reminder_text:
        if lang == "ar":
            base = "لم أتمكن من فهم نص التذكير. من فضلك حدد ما تريد أن يتم تذكيرك به ومتى."
        else:
            base = "I couldn't understand the reminder text. Please specify what you want to be reminded about and when."
        state["reply"] = base
        prof.pop("_reminder_text", None)
        prof.pop("_reminder_time", None)
        state["profile"] = prof
        return state
    
    if not reminder_time_str:
        if lang == "ar":
            base = "لم أتمكن من فهم وقت التذكير. من فضلك حدد الوقت (مثل: '14:30' أو 'بعد 30 دقيقة')."
        else:
            base = "I couldn't understand the reminder time. Please specify the time (e.g., '14:30' or 'in 30 minutes')."
        state["reply"] = base
        prof.pop("_reminder_text", None)
        prof.pop("_reminder_time", None)
        state["profile"] = prof
        return state
    
    # Get user timezone
    tzname = prof.get("tz") or "UTC"
    try:
        tz = ZoneInfo(tzname)
    except Exception:
        tz = ZoneInfo("UTC")
    
    now_local = datetime.now(tz)
    
    # Parse reminder time
    due_time: Optional[datetime] = None
    
    # Check if relative time (e.g., "in 30 minutes", "in 2 hours")
    relative_match = re.search(r"in\s+(\d+)\s*(minute|hour|min|hr|h)", reminder_time_str.lower())
    if relative_match:
        value = int(relative_match.group(1))
        unit = relative_match.group(2).lower()
        if unit in ("minute", "min"):
            due_time = now_local + timedelta(minutes=value)
        elif unit in ("hour", "hr", "h"):
            due_time = now_local + timedelta(hours=value)
    else:
        # Try to parse absolute time (HH:MM)
        time_match = re.search(r"(\d{1,2}):(\d{2})", reminder_time_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            due_time = datetime(now_local.year, now_local.month, now_local.day, hour, minute, tzinfo=tz)
            # If time has passed today, schedule for tomorrow
            if due_time <= now_local:
                due_time += timedelta(days=1)
    
    if not due_time:
        if lang == "ar":
            base = "لم أتمكن من فهم وقت التذكير. استخدم التنسيق: 'HH:MM' (مثل: '14:30') أو 'بعد X دقيقة/ساعة'."
        else:
            base = "I couldn't parse the reminder time. Use format: 'HH:MM' (e.g., '14:30') or 'in X minutes/hours'."
        state["reply"] = base
        prof.pop("_reminder_text", None)
        prof.pop("_reminder_time", None)
        state["profile"] = prof
        return state
    
    # Convert to UTC for Redis storage
    due_utc = due_time.astimezone(timezone.utc)
    due_utc_epoch = due_utc.timestamp()
    
    # Get user WhatsApp ID from state (should be set by handle_turn)
    wa_id = state.get("wa_id") or ""
    
    # Debug logging to help diagnose issues
    if not wa_id:
        print(f"[SCHED] Warning: wa_id not found in state. State keys: {list(state.keys())}")
        # Fallback: try to get from profile (shouldn't be needed, but just in case)
        wa_id = prof.get("_wa_id", "") or ""
    
    if wa_id:
        try:
            await enqueue_reminder(wa_id, reminder_text, due_utc_epoch)
            
            # Format confirmation message
            time_str = due_time.strftime("%H:%M")
            if lang == "ar":
                base = f"تم تحديد التذكير: '{reminder_text}' في الساعة {time_str}"
            else:
                base = f"Reminder set: '{reminder_text}' at {time_str}"
            
            state["reply"] = base
        except Exception as e:
            print(f"[SCHED] Failed to enqueue reminder: {e}")
            if lang == "ar":
                base = "حدث خطأ في تحديد التذكير. يرجى المحاولة مرة أخرى."
            else:
                base = "Failed to set reminder. Please try again."
            state["reply"] = base
    else:
        # If no wa_id, inform user (shouldn't happen in production)
        if lang == "ar":
            base = "لا يمكن تحديد التذكير بدون معرف المستخدم."
        else:
            base = "Cannot set reminder without user ID."
        state["reply"] = base
    
    # Clean up temporary slots
    prof.pop("_reminder_text", None)
    prof.pop("_reminder_time", None)
    state["profile"] = prof
    return state

async def calendar_connect(state: BotState) -> BotState:
    """Handle calendar connection request."""
    from data.db_postgres import set_zapier_mcp_url
    from mcp_client import is_calendar_connected
    
    q = state.get("question", "").strip()
    lang = _lang(state)
    wa_id = state.get("wa_id")
    
    if not wa_id:
        state["reply"] = "Cannot connect calendar without user ID." if lang != "ar" else "لا يمكن ربط التقويم بدون معرف المستخدم."
        return state
    
    # Check if already connected
    if await is_calendar_connected(wa_id):
        if lang == "ar":
            state["reply"] = "التقويم متصل بالفعل. يمكنك استخدام الأوامر المتاحة."
        else:
            state["reply"] = "Calendar is already connected. You can use available commands."
        return state
    
    # Extract MCP URL from user message
    # Look for URL pattern or ask user to provide it
    url_pattern = r"https://mcp\.zapier\.com/api/mcp/s/[^\s]+"
    match = re.search(url_pattern, q)
    
    if match:
        mcp_url = match.group(0)
        try:
            await set_zapier_mcp_url(wa_id, mcp_url)
            if lang == "ar":
                state["reply"] = "تم ربط التقويم بنجاح! يمكنك الآن إنشاء وعرض الأحداث."
            else:
                state["reply"] = "Calendar connected successfully! You can now create and view events."
        except Exception as e:
            if lang == "ar":
                state["reply"] = f"فشل ربط التقويم: {str(e)}"
            else:
                state["reply"] = f"Failed to connect calendar: {str(e)}"
    else:
        if lang == "ar":
            state["reply"] = "يرجى إرسال رابط Zapier MCP الخاص بك. مثال: https://mcp.zapier.com/api/mcp/s/..."
        else:
            state["reply"] = "Please send your Zapier MCP server URL. Example: https://mcp.zapier.com/api/mcp/s/..."
    
    return state

async def calendar_create_event(state: BotState) -> BotState:
    """Create a calendar event."""
    from mcp_client import call_calendar_tool, is_calendar_connected
    
    lang = _lang(state)
    wa_id = state.get("wa_id")
    
    if not wa_id:
        state["reply"] = "Cannot create event without user ID." if lang != "ar" else "لا يمكن إنشاء حدث بدون معرف المستخدم."
        return state
    
    if not await is_calendar_connected(wa_id):
        if lang == "ar":
            state["reply"] = "التقويم غير متصل. يرجى الاتصال أولاً باستخدام أمر 'connect calendar'."
        else:
            state["reply"] = "Calendar not connected. Please connect first using 'connect calendar'."
        return state
    
    # Parse event details from question
    q = state.get("question", "")
    
    # Use quick_add_event as it's simpler and can parse natural language
    result = await call_calendar_tool(
        wa_id,
        "google_calendar_quick_add_event",
        {
            "instructions": "Create an event from the user's text",
            "text": q,
            "calendarid": "primary"  # Use primary calendar
        }
    )
    
    if result["success"]:
        if lang == "ar":
            state["reply"] = "تم إنشاء الحدث بنجاح!"
        else:
            state["reply"] = "Event created successfully!"
    else:
        state["reply"] = result["error"] or ("فشل إنشاء الحدث." if lang == "ar" else "Failed to create event.")
    
    return state

async def calendar_view_events(state: BotState) -> BotState:
    """View upcoming calendar events."""
    from mcp_client import call_calendar_tool, is_calendar_connected
    
    lang = _lang(state)
    wa_id = state.get("wa_id")
    
    if not wa_id:
        state["reply"] = "Cannot view events without user ID." if lang != "ar" else "لا يمكن عرض الأحداث بدون معرف المستخدم."
        return state
    
    if not await is_calendar_connected(wa_id):
        if lang == "ar":
            state["reply"] = "التقويم غير متصل. يرجى الاتصال أولاً."
        else:
            state["reply"] = "Calendar not connected. Please connect first."
        return state
    
    # Find events for today and next few days
    from datetime import datetime, timedelta
    now = datetime.now()
    end_time = (now + timedelta(days=7)).isoformat()
    
    result = await call_calendar_tool(
        wa_id,
        "google_calendar_find_events",
        {
            "instructions": "Find upcoming events",
            "calendarid": "primary",
            "start_time": now.isoformat(),
            "end_time": end_time,
            "ordering": "startTime"
        }
    )
    
    if result["success"]:
        events = result["data"]
        if isinstance(events, list) and len(events) > 0:
            event_list = []
            for event in events[:10]:  # Show up to 10 events
                summary = event.get("summary", "No title")
                start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "Unknown time")
                event_list.append(f"- {summary} ({start})")
            
            reply = "Upcoming events:\n" + "\n".join(event_list)
            if lang == "ar":
                reply = "الأحداث القادمة:\n" + "\n".join(event_list)
            state["reply"] = reply
        else:
            if lang == "ar":
                state["reply"] = "لا توجد أحداث قادمة."
            else:
                state["reply"] = "No upcoming events found."
    else:
        state["reply"] = result["error"] or ("فشل استرجاع الأحداث." if lang == "ar" else "Failed to retrieve events.")
    
    return state

async def calendar_find_events(state: BotState) -> BotState:
    """Search for events in calendar."""
    from mcp_client import call_calendar_tool, is_calendar_connected
    
    lang = _lang(state)
    wa_id = state.get("wa_id")
    
    if not await is_calendar_connected(wa_id):
        if lang == "ar":
            state["reply"] = "التقويم غير متصل. يرجى الاتصال أولاً."
        else:
            state["reply"] = "Calendar not connected. Please connect first."
        return state
    
    # Parse search query from question
    q = state.get("question", "")
    
    # Find events with search parameters
    from datetime import datetime, timedelta
    now = datetime.now()
    end_time = (now + timedelta(days=30)).isoformat()  # Search in next 30 days
    
    result = await call_calendar_tool(
        wa_id,
        "google_calendar_find_events",
        {
            "instructions": f"Search for events matching: {q}",
            "calendarid": "primary",
            "start_time": now.isoformat(),
            "end_time": end_time,
            "ordering": "startTime"
        }
    )
    
    if result["success"]:
        events = result["data"]
        if isinstance(events, list) and len(events) > 0:
            event_list = []
            for event in events[:10]:
                summary = event.get("summary", "No title")
                start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date", "Unknown")
                event_list.append(f"- {summary} ({start})")
            
            reply = f"Found {len(events)} events:\n" + "\n".join(event_list)
            if lang == "ar":
                reply = f"تم العثور على {len(events)} حدث:\n" + "\n".join(event_list)
            state["reply"] = reply
        else:
            if lang == "ar":
                state["reply"] = "لم يتم العثور على أحداث."
            else:
                state["reply"] = "No events found."
    else:
        state["reply"] = result["error"] or ("فشل البحث عن الأحداث." if lang == "ar" else "Failed to search events.")
    
    return state

async def calendar_delete_event(state: BotState) -> BotState:
    """Delete a calendar event."""
    from mcp_client import call_calendar_tool, is_calendar_connected
    
    lang = _lang(state)
    wa_id = state.get("wa_id")
    
    if not await is_calendar_connected(wa_id):
        if lang == "ar":
            state["reply"] = "التقويم غير متصل. يرجى الاتصال أولاً."
        else:
            state["reply"] = "Calendar not connected. Please connect first."
        return state
    
    # Extract event ID from question
    # Try to extract event ID from message - this is a simplified version
    q = state.get("question", "")
    
    # For now, we'll need the user to provide the event ID
    # A better implementation would first list events and let user select
    if not q or "event" not in q.lower():
        if lang == "ar":
            state["reply"] = "يرجى تحديد معرف الحدث المراد حذفه. استخدم 'view events' لرؤية الأحداث أولاً."
        else:
            state["reply"] = "Please specify the event ID to delete. Use 'view events' to see events first."
        return state
    
    # Note: This is a simplified implementation
    # In production, you'd want to first find events and let user select which one to delete
    # For now, we'll inform the user they need to provide the event ID
    if lang == "ar":
        state["reply"] = "لحذف حدث، يرجى استخدام معرف الحدث. استخدم 'view events' لرؤية الأحداث ومعرفاتها."
    else:
        state["reply"] = "To delete an event, please use the event ID. Use 'view events' to see events and their IDs."
    
    return state

async def general(state: BotState) -> BotState:
    """
    Handles general questions by politely redirecting users to specialized features.
    Prevents hallucinations by not using LLM for general chat.
    """
    lang = _lang(state)
    
    if lang == "ar":
        reply = (
            "أعتذر، أنا متخصص في:\n"
            "• أوقات الصلاة (الفجر، الظهر، العصر، المغرب، العشاء)\n"
            "• التاريخ الهجري والميلادي\n"
            "• الصلاة القادمة\n"
            "• التذكيرات والتنبيهات\n"
            "• إدارة التقويم (ربط، إنشاء، عرض، حذف الأحداث)\n\n"
            "من فضلك اسألني عن إحدى هذه الخدمات."
        )
    else:
        reply = (
            "I'm a specialized assistant for Islamic prayer services.\n\n"
            "I can help you with:\n"
            "• Prayer times (Fajr, Dhuhr, Asr, Maghrib, Isha)\n"
            "• Islamic (Hijri) and Gregorian dates\n"
            "• Next prayer time\n"
            "• Setting reminders\n"
            "• Calendar management (connect, create, view, delete events)\n\n"
            "Please ask me about one of these services."
        )
    
    state["reply"] = reply
    return state

async def noop(state: BotState) -> BotState:
    """Sink node used while onboarding is in progress."""
    prof = state.get("profile", {}) or {}
    if prof.get("_onboarding_ack"):
        prof.pop("_onboarding_ack", None)
    if _profile_complete(prof) and prof.get("_await"):
        prof.pop("_await", None)
    state["profile"] = prof
    return state

# -------------------------
# Graph
# -------------------------
workflow = StateGraph(BotState)
workflow.add_node("ensure_profile", ensure_profile)
workflow.add_node("classify", classify)
workflow.add_node("islamic_date", islamic_date)
workflow.add_node("prayer_times", prayer_times)
workflow.add_node("next_prayer", next_prayer)
workflow.add_node("scheduler_agent", scheduler_agent)
workflow.add_node("calendar_connect", calendar_connect)
workflow.add_node("calendar_create_event", calendar_create_event)
workflow.add_node("calendar_view_events", calendar_view_events)
workflow.add_node("calendar_find_events", calendar_find_events)
workflow.add_node("calendar_delete_event", calendar_delete_event)
workflow.add_node("general", general)
workflow.add_node("noop", noop)

workflow.set_entry_point("ensure_profile")

def route_after_profile(state: BotState):
    prof = state.get("profile", {}) or {}
    if prof.get("_await") or not _profile_complete(prof) or prof.get("_confirm_loc"):
        return "noop"
    return "classify"

workflow.add_conditional_edges("ensure_profile", route_after_profile, {
    "noop": "noop",
    "classify": "classify"
})

def route(state: BotState):
    return state["intent"]

workflow.add_conditional_edges("classify", route, {
    "islamic_date": "islamic_date",
    "prayer_times": "prayer_times",
    "next_prayer": "next_prayer",
    "reminder": "scheduler_agent",
    "calendar_connect": "calendar_connect",
    "calendar_create_event": "calendar_create_event",
    "calendar_view_events": "calendar_view_events",
    "calendar_find_events": "calendar_find_events",
    "calendar_delete_event": "calendar_delete_event",
    "general": "general",
})

for node in ["islamic_date","prayer_times","next_prayer","scheduler_agent","calendar_connect","calendar_create_event","calendar_view_events","calendar_find_events","calendar_delete_event","general","noop"]:
    workflow.add_edge(node, END)

app_graph = workflow.compile()

# -------------------------
# Runner for local testing
# -------------------------
async def main():
    profile: Dict[str, str] = {}
    context: Dict[str, Any] = {"short_history": [], "semantic_snippets": []}
    print("Assalamualaikum! I'll collect a few details to personalize timings.")
    try:
        while True:
            q = input("You: ")
            if q.lower() in {"quit", "exit"}:
                break
            # auto-set language each turn for CLI
            profile = _auto_set_lang(profile, q)
            # keep a tiny local history for demo CLI
            context["short_history"].append(("user", q))
            out = await app_graph.ainvoke({"question": q, "profile": profile, "context": context})
            profile = out.get("profile", profile)
            reply = out.get("reply", "")
            context["short_history"].append(("assistant", reply))
            print("Bot:", reply)
    finally:
        try:
            await HTTP.aclose()
        except Exception:
            pass

# --- Entry for server usage ---
async def handle_turn(
    question: str,
    profile: Dict[str, str],
    context: Optional[Dict[str, Any]] = None,
    wa_id: Optional[str] = None
) -> tuple[str, Dict[str, str]]:
    """
    Runs one turn through the graph and returns (reply_text, new_profile).
    Server may pass profile['lang'] = 'ar' or 'en'; if not, auto-detected from `question`.
    `context` may contain:
      - short_history: list[(role, text)]   # chronological
      - semantic_snippets: list[str]
    `wa_id` is the WhatsApp user ID, needed for reminder scheduling.
    """
    profile = _auto_set_lang(profile or {}, question or "")
    state_dict = {"question": question, "profile": profile, "context": context or {}}
    if wa_id:
        state_dict["wa_id"] = wa_id
        # wa_id set successfully
    else:
        print(f"[WARN] handle_turn called without wa_id - reminders will not work")
    result = await app_graph.ainvoke(state_dict)
    reply = (result.get("reply") or "").strip()
    new_profile = result.get("profile", profile or {})
    return reply, new_profile

if __name__ == "__main__":
    print(mermaid_diagram())
    asyncio.run(main())
