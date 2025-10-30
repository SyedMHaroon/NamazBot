"""
Simple LangGraph bot for local testing — with onboarding.
- Collects: Name, Email, Location ("City - Country")
- Routes via LLM classifier (structured JSON): islamic_date | prayer_times | next_prayer | general
- Validates location against Aladhan
- Uses Gemini (GEMINI_API_KEY) for classification + general answers
- Language-aware replies: Arabic if profile["lang"] == "ar", else English
- Auto-detects Arabic per turn if profile["lang"] not explicitly set
- No Flask/FastAPI — runs in terminal
"""

import os
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Literal, TypedDict, Optional, Any
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

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

INTENT_LABELS = ["islamic_date", "prayer_times", "next_prayer", "general"]
PRAYER_NAMES  = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]
PRAYER_ORDER  = ["Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]

# -------------------------
# State definition
# -------------------------
class BotState(TypedDict, total=False):
    question: str
    intent: Literal["islamic_date", "prayer_times", "next_prayer", "general"]
    profile: Dict[str, str]   # name, email, city, country, _await?, _setup_done?, _onboarding_ack?, _requested_prayer?, _requested_date?, lang?
    reply: str

# -------------------------
# Helpers: API + parsing
# -------------------------
RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

async def aladhan_fetch(city: Optional[str], country: Optional[str], date: Optional[str] = None) -> Dict[str, Any]:
    """
    If both city and country are present -> use timingsByCity.
    If only city is present -> use timingsByAddress (no guessing a country).
    """
    if not city:
        raise ValueError("aladhan_fetch requires at least a city.")

    base_params = {"method": 2}
    if date:
        base_params["date"] = date

    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as c:
        if city and country:
            url = "https://api.aladhan.com/v1/timingsByCity"
            params = {**base_params, "city": city, "country": country}
        else:
            url = "https://api.aladhan.com/v1/timingsByAddress"
            params = {**base_params, "address": city}

        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()["data"]

async def aladhan(city: str, country: str, date: Optional[str] = None) -> Dict[str, Any]:
    url = "https://api.aladhan.com/v1/timingsByCity"
    params = {"city": city, "country": country, "method": 2}
    if date:
        params["date"] = date
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        return r.json()["data"]

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

async def llm_intent_json(question: str) -> dict:
    """
    Ask Gemini for strict JSON: {"intent": "...", "slots": {"prayer_name":..., "date":..., "city":..., "country":...}}
    IMPORTANT: The model must NOT infer country; only set it if the user explicitly typed it.
    """
    system = (
        "You are a router that ONLY returns strict JSON. No prose, no markdown.\n"
        "Allowed intents: islamic_date, prayer_times, next_prayer, general.\n"
        "Slots (all optional):\n"
        "  - prayer_name: Fajr|Dhuhr|Asr|Maghrib|Isha\n"
        "  - date: today|tomorrow|YYYY-MM-DD\n"
        "  - city: string (ONLY if user typed a city)\n"
        "  - country: string (ONLY if user explicitly typed a country; DO NOT GUESS OR INFER)\n"
        "If unsure about ANY slot, set it to null. DO NOT invent or infer countries.\n"
        "Respond with exactly this JSON schema:\n"
        "{\n"
        "  \"intent\": \"islamic_date|prayer_times|next_prayer|general\",\n"
        "  \"slots\": {\"prayer_name\": null|string, \"date\": null|string, \"city\": null|string, \"country\": null|string}\n"
        "}\n"
    )
    prompt = f"{system}\nUser: {question}\n"
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
            profile.pop("_staged_loc", None)
            profile.pop("_confirm_loc", None)
            profile.pop("_await", None)
            profile["_setup_done"] = True
            state["reply"] = (
                f"Shukriya, {profile.get('name', '')}! Setup complete for "
                f"{profile['city']}, {profile['country']}.\n"
                "You can now ask: 'Fajr time', 'Next prayer', or 'Islamic date'."
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
        state["reply"] = (
            f"Shukriya, {profile['name']}! Setup complete for "
            f"{profile['city']}, {profile['country']}.\n"
            "You can now ask: 'Fajr time', 'Next prayer', or 'Islamic date'."
        )

    state["profile"] = profile
    return state

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

# <<< NEW >>> Auto-detect language per turn if not supplied
def _auto_set_lang(profile: Dict[str, str], question: str) -> Dict[str, str]:
    """
    Returns a shallow copy of profile with 'lang' set from the question if not explicitly set for this turn.
    Arabic letters ⇒ 'ar', else 'en'.
    """
    prof = dict(profile or {})
    # Always refresh to the question's language so it adapts turn-by-turn.
    prof["lang"] = "ar" if _has_ar(question) else "en"
    return prof

# -------------------------
# Intent classification (LLM router)
# -------------------------
async def classify(state: BotState) -> BotState:
    prof = state.get("profile", {}) or {}
    if prof.get("_await") or not _profile_complete(prof) or prof.get("_onboarding_ack") or prof.get("_confirm_loc"):
        state["intent"] = "general"
        return state

    q = state.get("question", "") or ""
    data = await llm_intent_json(q)
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

    state["profile"] = prof
    state["intent"] = label
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
    classify -->|general| general
    islamic_date --> END((END))
    prayer_times --> END
    next_prayer --> END
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
    tzname = d.get("meta", {}).get("timezone", "UTC")
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

async def general(state: BotState) -> BotState:
    q = state.get("question", "")
    lang = _lang(state)

    if lang == "ar":
        system = (
            "أجب باللغة العربية الفصحى فقط. لا تستخدم الإنجليزية. "
            "قدّم إجابات موجزة ودقيقة، وحافظ على مصطلحات الصلاة والتاريخ الهجري كما هي."
        )
    else:
        system = (
            "Answer only in English. "
            "Be concise and accurate; keep Islamic terms (e.g., prayer names, Hijri) consistent."
        )

    prompt = f"{system}\n\nUser: {q}"
    try:
        res = await llm.ainvoke(prompt)
        text = res.content if hasattr(res, "content") else str(res)
        state["reply"] = text.strip()
    except Exception as e:
        state["reply"] = f"Error generating response: {e}"
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
    "general": "general",
})

for node in ["islamic_date","prayer_times","next_prayer","general","noop"]:
    workflow.add_edge(node, END)

app_graph = workflow.compile()

# -------------------------
# Runner for local testing (stateful profile across turns)
# -------------------------
async def main():
    profile: Dict[str, str] = {}
    print("Assalamualaikum! I'll collect a few details to personalize timings.")
    while True:
        q = input("You: ")
        if q.lower() in {"quit", "exit"}:
            break
        # <<< NEW >>> auto-set language each turn for CLI
        profile = _auto_set_lang(profile, q)
        out = await app_graph.ainvoke({"question": q, "profile": profile})
        profile = out.get("profile", profile)
        print("Bot:", out.get("reply", ""))

# --- Convenience entrypoint for webhooks ---
async def handle_turn(question: str, profile: Dict[str, str]) -> tuple[str, Dict[str, str]]:
    """
    Runs one turn through the graph and returns (reply_text, new_profile).
    Server may pass profile['lang'] = 'ar' or 'en'; if not, it will be auto-detected from `question`.
    """
    # <<< NEW >>> auto-set language each turn for server usage too
    profile = _auto_set_lang(profile or {}, question or "")
    result = await app_graph.ainvoke({"question": question, "profile": profile})
    reply = (result.get("reply") or "").strip()
    new_profile = result.get("profile", profile or {})
    return reply, new_profile

if __name__ == "__main__":
    print(mermaid_diagram())
    asyncio.run(main())
