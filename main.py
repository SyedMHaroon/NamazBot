"""
Simple LangGraph bot for local testing — with onboarding.
- Collects: Name, Email, Location ("City - Country")
- Routes via LLM classifier (structured JSON): islamic_date | prayer_times | next_prayer | general
- Validates location against Aladhan
- Uses Gemini (GEMINI_API_KEY) for classification + general answers
- No Flask/FastAPI — runs in terminal
"""

import os
import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Literal, TypedDict, Optional, Any, Tuple
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
    profile: Dict[str, str]   # name, email, city, country, _await?, _setup_done?, _onboarding_ack?, _requested_prayer?, _requested_date?
    reply: str

# -------------------------
# Helpers: API + parsing
# -------------------------
RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

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
    # quick alias expansions
    aliases = {
        "KSA": "Saudi Arabia", "UAE": "United Arab Emirates",
        "UK": "United Kingdom", "USA": "United States",
        "US": "United States", "PK": "Pakistan",
        "Ksa": "Saudi Arabia", "Uae": "United Arab Emirates",
        "Uk": "United Kingdom", "Usa": "United States", "Pk": "Pakistan",
    }
    if n in aliases:
        return aliases[n]
    # try exact name match
    try:
        for c in pycountry.countries:
            if n.lower() == c.name.lower():
                return c.name
            # also check common/official names if present
            if hasattr(c, "common_name") and n.lower() == c.common_name.lower():
                return c.common_name
        # fuzzy: startswith
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

    # basic syntactic sanity
    if len(city_raw) < 2 or len(country_raw) < 2:
        return None, None

    # normalize
    city = city_raw.title()
    country_norm = normalize_country_name(country_raw)
    if not country_norm:
        return city, None  # city might be fine but country unknown
    return city, country_norm

async def validate_location_against_api(city: str, country: str) -> tuple[bool, Optional[str]]:
    """
    Probe Aladhan. Return (ok, timezone). ok only if timings exist AND a timezone is present.
    """
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
    """
    system = (
        "You are a router that ONLY returns strict JSON. No prose, no markdown.\n"
        "Allowed intents: islamic_date, prayer_times, next_prayer, general.\n"
        "Slots (all optional): prayer_name (Fajr|Dhuhr|Asr|Maghrib|Isha), "
        "date (today|tomorrow|YYYY-MM-DD), city (string), country (string).\n"
        "If unsure about a slot, set it to null.\n"
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
    # normalize prayer name
    pn = slots.get("prayer_name")
    if pn:
        mapping = {"fajr":"Fajr","zuhr":"Dhuhr","dhuhr":"Dhuhr","asr":"Asr","maghrib":"Maghrib","isha":"Isha"}
        slots["prayer_name"] = mapping.get(str(pn).strip().lower(), None)
        if slots["prayer_name"] not in PRAYER_NAMES:
            slots["prayer_name"] = None
    # normalize city/country
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

            # mark done immediately
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

        # Stage and ask confirmation — clear _await first
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

    # --- Final confirmation message once all info present ---
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
# Intent classification (LLM router)
# -------------------------
async def classify(state: BotState) -> BotState:
    prof = state.get("profile", {}) or {}
    # Safety: if onboarding isn't complete (or ack just printed), don't route
    if prof.get("_await") or not _profile_complete(prof) or prof.get("_onboarding_ack"):
        state["intent"] = "general"  # not used; route_after_profile blocks classify anyway
        return state

    q = state.get("question", "")
    data = await llm_intent_json(q)
    label = data["intent"]
    slots = data.get("slots", {}) or {}

    # Inline overrides
    if slots.get("city"):
        prof["city"] = slots["city"]
    if slots.get("country"):
        prof["country"] = slots["country"]
    state["profile"] = prof

    if slots.get("prayer_name"):
        state["profile"]["_requested_prayer"] = slots["prayer_name"]
    else:
        state["profile"].pop("_requested_prayer", None)

    if slots.get("date"):
        state["profile"]["_requested_date"] = slots["date"]
    else:
        state["profile"].pop("_requested_date", None)

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
# Task nodes
# -------------------------
async def islamic_date(state: BotState) -> BotState:
    city    = state["profile"]["city"]
    country = state["profile"]["country"]
    d = await aladhan(city, country)
    hijri = d["date"]["hijri"]["date"]
    greg  = d["date"]["readable"]
    state["reply"] = f"Islamic (Hijri) date in {city}, {country}: {hijri}\nGregorian: {greg}"
    return state

async def prayer_times(state: BotState) -> BotState:
    city    = state["profile"]["city"]
    country = state["profile"]["country"]

    # date support: today (default). If 'tomorrow' or YYYY-MM-DD requested, use it.
    date_req = state["profile"].get("_requested_date")
    date_param: Optional[str] = None
    if date_req:
        s = date_req.strip().lower()
        if s == "today":
            date_param = None
        elif s == "tomorrow":
            tzname = (await aladhan(city, country)).get("meta", {}).get("timezone", "UTC")
            now = datetime.now(ZoneInfo(tzname))
            date_param = (now + timedelta(days=1)).strftime("%d-%m-%Y")
        else:
            try:
                dt = datetime.strptime(date_req, "%Y-%m-%d")
                date_param = dt.strftime("%d-%m-%Y")
            except Exception:
                date_param = None

    d = await aladhan(city, country, date_param)
    t = {k: clean_time(v) for k, v in d["timings"].items()}
    req = state["profile"].get("_requested_prayer")

    if req in PRAYER_NAMES:
        state["reply"] = f"{req} time in {city}, {country}: {t.get(req, 'N/A')}"
        return state

    lines = [f"{k}: {t.get(k, 'N/A')}" for k in PRAYER_ORDER]
    when = "today" if not date_param else (state["profile"].get("_requested_date") or "the selected date")
    state["reply"] = f"Prayer times {when} for {city}, {country}:\n" + "\n".join(lines)
    return state

async def next_prayer(state: BotState) -> BotState:
    city    = state["profile"]["city"]
    country = state["profile"]["country"]
    d = await aladhan(city, country)
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
        # after Isha → fetch tomorrow's Fajr
        tomorrow = (now + timedelta(days=1)).strftime("%d-%m-%Y")
        d2 = await aladhan(city, country, tomorrow)
        fajr = clean_time(d2["timings"]["Fajr"])
        h, m = map(int, fajr.split(":"))
        nxt_name = "Fajr"
        nxt_time = datetime(now.year, now.month, now.day, h, m, tzinfo=tz) + timedelta(days=1)

    rem = nxt_time - now
    hours, rem_mins = divmod(int(rem.total_seconds()) // 60, 60)
    state["reply"] = f"Next prayer in {city}, {country}: {nxt_name} at {nxt_time.strftime('%H:%M')} ({hours}h {rem_mins}m left)"
    return state

async def general(state: BotState) -> BotState:
    q = state.get("question", "")
    try:
        res = await llm.ainvoke(q)
        state["reply"] = res.content if hasattr(res, "content") else str(res)
    except Exception as e:
        state["reply"] = f"Error generating response: {e}"
    return state

async def noop(state: BotState) -> BotState:
    """
    Sink node used while onboarding is in progress.
    Ends the current turn. It should NOT clear confirmation flags,
    otherwise the staged location is lost before the user can say Yes/No.
    """
    prof = state.get("profile", {}) or {}

    # Clear only the one-turn 'thank you' flag so the next turn can route normally.
    if prof.get("_onboarding_ack"):
        prof.pop("_onboarding_ack", None)

    # DO NOT clear _confirm_loc or _staged_loc here.
    # If we cleared them, the confirmation step would be broken.

    # Defensive: if profile is already complete but _await is somehow set, drop it.
    if _profile_complete(prof) and prof.get("_await"):
        prof.pop("_await", None)

    state["profile"] = prof
    # No reply on purpose.
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
    # Only block routing if we're still collecting a field or confirming location
    if prof.get("_await") or not _profile_complete(prof) or prof.get("_confirm_loc"):
        return "noop"     # end this turn; don't run classify/LLM yet
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
        out = await app_graph.ainvoke({"question": q, "profile": profile})
        profile = out.get("profile", profile)  # persist across turns
        print("Bot:", out.get("reply", ""))
        # Debug (optional):
        # print("[debug] intent:", out.get("intent"), "| profile:", profile)

if __name__ == "__main__":
    print(mermaid_diagram())     # copy into mermaid.live to visualize the flow
    asyncio.run(main())
