import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import httpx
import os
# Redis access (async)
from data.redis_store import get_redis
# WhatsApp send
from wa_client import send_whatsapp_text
from dotenv import load_dotenv
load_dotenv()

# Digest schedule and deduplication settings
DIGEST_HOUR = int(os.getenv("DIGEST_HOUR"))
DIGEST_MINUTE = int(os.getenv("DIGEST_MINUTE"))
DIGEST_DEDUPE = os.getenv("DIGEST_DEDUPE")

# ---------------- Region-aware Aladhan helpers ----------------
def _choose_method(country: str) -> int:
    c = (country or "").lower()
    if any(x in c for x in ["pakistan", "karachi"]):
        return 1  # University of Islamic Sciences, Karachi
    if any(x in c for x in ["canada", "usa", "united states", "america"]):
        return 2  # ISNA
    if any(x in c for x in ["uk", "england", "britain", "london", "europe"]):
        return 3  # Muslim World League
    if any(x in c for x in ["saudi", "arab", "uae", "qatar", "oman", "bahrain"]):
        return 4  # Umm al-Qura
    return 2  # default fallback = ISNA


async def _aladhan_by_city(city: str, country: str, date_str: Optional[str]) -> Dict[str, Any]:
    """
    Use the *dated* endpoint to avoid 302s and select method dynamically.
    """
    method_id = _choose_method(country)
    base = "https://api.aladhan.com/v1/timingsByCity"
    path = f"/{date_str}" if date_str else ""
    url = f"{base}{path}"
    params = {
        "city": city,
        "country": country,
        "method": method_id,
        "latitudeAdjustmentMethod": 3,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(25.0), follow_redirects=True) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return (r.json() or {}).get("data", {}) or {}


def _hhmm(s: str) -> str:
    import re
    m = re.search(r"(\d{1,2}:\d{2})", s or "")
    return m.group(1) if m else (s or "N/A")


# ------------- Message builders -------------
async def build_date_only_message(city: str, country: str, lang: str, date_str: str) -> str:
    d = await _aladhan_by_city(city, country, date_str)
    hijri = (d.get("date", {}).get("hijri", {}) or {}).get("date", "")
    greg = (d.get("date", {}) or {}).get("readable", "")
    if (lang or "en").lower() == "ar":
        return f"التاريخ الهجري في {city}, {country}: {hijri}\nالتاريخ الميلادي: {greg}"
    return f"Islamic (Hijri) date in {city}, {country}: {hijri}\nGregorian: {greg}"


async def build_digest_message(city: str, country: str, lang: str, date_str: str) -> str:
    """
    Build bilingual digest message with both Arabic and English.
    This ensures all users can understand the digest regardless of language preference.
    """
    d = await _aladhan_by_city(city, country, date_str)
    hijri = (d.get("date", {}).get("hijri", {}) or {}).get("date", "")
    greg = (d.get("date", {}) or {}).get("readable", "")
    t = {k: _hhmm(v) for k, v in (d.get("timings", {}) or {}).items()}

    # Determine appropriate greeting based on hour
    current_hour = datetime.now().hour
    
    # Arabic greeting
    if current_hour < 12:
        greeting_ar = "صباح الخير!"
        greeting_en = "Good morning!"
    elif current_hour < 17:
        greeting_ar = "مساء الخير!"
        greeting_en = "Good afternoon!"
    else:
        greeting_ar = "مساء الخير!"
        greeting_en = "Good evening!"
    
    # Build bilingual message
    lines = [
        f"{greeting_ar} / {greeting_en}",
        f"📍 {city}, {country}",
        "",
        f"التاريخ الهجري / Hijri date: {hijri}",
        f"التاريخ الميلادي / Gregorian: {greg}",
        "",
        "أوقات الصلاة اليوم / Today's prayer times:",
        f"الفجر / Fajr: {t.get('Fajr','N/A')}",
        f"الظهر / Dhuhr: {t.get('Dhuhr','N/A')}",
        f"العصر / Asr: {t.get('Asr','N/A')}",
        f"المغرب / Maghrib: {t.get('Maghrib','N/A')}",
        f"العشاء / Isha: {t.get('Isha','N/A')}",
    ]
    return "\n".join(lines)


# ------------- Subscription helpers -------------
async def subscribe_to_digest(wa_id: str) -> bool:
    """Add user to digest subscribers list."""
    r = await get_redis()
    try:
        await r.sadd("digest:subs", wa_id)
        print(f"[SCHED] Subscribed {wa_id} to daily digest")
        return True
    except Exception as e:
        print(f"[SCHED] Failed to subscribe {wa_id}: {e}")
        return False

async def unsubscribe_from_digest(wa_id: str) -> bool:
    """Remove user from digest subscribers list."""
    r = await get_redis()
    try:
        await r.srem("digest:subs", wa_id)
        print(f"[SCHED] Unsubscribed {wa_id} from daily digest")
        return True
    except Exception as e:
        print(f"[SCHED] Failed to unsubscribe {wa_id}: {e}")
        return False

async def is_subscribed_to_digest(wa_id: str) -> bool:
    """Check if user is subscribed to digest."""
    r = await get_redis()
    try:
        return await r.sismember("digest:subs", wa_id)
    except Exception:
        return False

# ------------- Scheduled ticks -------------
async def run_digest_tick(get_profile, send_text=send_whatsapp_text, hour: int = DIGEST_HOUR, minute: int = DIGEST_MINUTE, dedupe: bool = DIGEST_DEDUPE):
    """
    Iterate subscribers in Redis set 'digest:subs'.
    If local time ≈ hour:minute (±1 min) and not sent today, push the digest.
    
    Args:
        dedupe: If True, skip if already sent today. If False, always send (for testing).
        hour: The hour of the day to send the digest.
        minute: The minute of the hour to send the digest.
    """
    import zoneinfo
    r = await get_redis()
    wa_ids = await r.smembers("digest:subs")
    if not wa_ids:
        # Debug: log when no subscribers found
        print(f"[SCHED] No subscribers found in digest:subs set")
        return

    print(f"[SCHED] Checking {len(wa_ids)} subscribers for digest at {hour:02d}:{minute:02d} (dedupe={dedupe})")
    
    for wa_id in wa_ids:
        try:
            profile = await get_profile(wa_id) or {}
            city = (profile.get("city") or "").strip()
            country = (profile.get("country") or "").strip()
            tz_name = (profile.get("tz") or "").strip()
            lang = (profile.get("lang") or "en").lower()
            if not (city and country and tz_name):
                print(f"[SCHED] Skipping {wa_id}: missing profile data (city={bool(city)}, country={bool(country)}, tz={bool(tz_name)})")
                continue

            try:
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception as tz_err:
                print(f"[SCHED] Skipping {wa_id}: invalid timezone '{tz_name}': {tz_err}")
                continue

            now_local = datetime.now(tz)
            current_minutes = now_local.hour * 60 + now_local.minute
            target_minutes = hour * 60 + minute
            delta_minutes = abs(current_minutes - target_minutes)
            
            if delta_minutes > 1:
                # Only log occasionally to avoid spam
                if now_local.minute % 10 == 0:  # Log every 10 minutes
                    print(f"[SCHED] {wa_id} time check: current={now_local.strftime('%H:%M')}, target={hour:02d}:{minute:02d}, delta={delta_minutes}min")
                continue

            sent_key = f"digest:sent:{wa_id}:{now_local.date().isoformat()}"
            
            # Check deduplication only if dedupe=True
            if dedupe and await r.get(sent_key):
                print(f"[SCHED] Skipping {wa_id}: already sent digest today ({now_local.date().isoformat()})")
                continue

            date_str = now_local.strftime("%d-%m-%Y")
            print(f"[SCHED] Sending digest to {wa_id} at {now_local.strftime('%H:%M')} ({tz_name}) [dedupe={dedupe}]")
            # Build bilingual message (both Arabic and English)
            msg = await build_digest_message(city, country, lang, date_str)
            await send_text(wa_id, msg)
            
            # Only set sent key if dedupe is enabled
            if dedupe:
                await r.set(sent_key, "1", ex=36 * 3600)
            
            print(f"[SCHED] Successfully sent digest to {wa_id}")

        except Exception as e:
            print(f"[SCHED] digest failed for {wa_id}:", e)


# ---- reminders ----
REM_ZSET = "reminders:zset"  # member = JSON, score = UTC epoch seconds


async def enqueue_reminder(wa_id: str, text: str, due_utc_epoch: float, meta: Optional[dict] = None):
    r = await get_redis()
    payload = {"wa_id": wa_id, "text": text, "due_utc": due_utc_epoch}
    if meta:
        payload["meta"] = meta
    await r.zadd(REM_ZSET, {json.dumps(payload): due_utc_epoch})


async def run_reminder_tick(send_text=send_whatsapp_text, batch_window_seconds: int = 60):
    """
    Every minute: pop all reminders with due_utc <= now.
    """
    r = await get_redis()
    now = datetime.now(timezone.utc).timestamp() + 1  # small buffer
    items: List[bytes] = await r.zrangebyscore(REM_ZSET, min=0, max=now)
    if not items:
        return
    await r.zrem(REM_ZSET, *items)
    for raw in items:
        try:
            data = json.loads(raw)
            await send_text(data["wa_id"], f"⏰ Reminder: {data['text']}")
        except Exception as e:
            print("[SCHED] reminder delivery failed:", e)


# ---- prayer reminders (10 minutes before each prayer) ----
async def run_prayer_reminder_tick(get_profile, send_text=send_whatsapp_text):
    """
    Every minute: Check all users and send prayer reminders 10 minutes before each prayer time.
    Uses deduplication to ensure each prayer reminder is sent only once per day.
    """
    import zoneinfo
    from main import aladhan_fetch, PRAYER_ORDER, PRAYER_NAMES
    
    r = await get_redis()
    
    # Get all users (could be from digest subscribers or all profiles)
    # For now, we'll check digest subscribers as they have complete profiles
    wa_ids = await r.smembers("digest:subs")
    if not wa_ids:
        return
    
    for wa_id in wa_ids:
        try:
            profile = await get_profile(wa_id) or {}
            city = (profile.get("city") or "").strip()
            country = (profile.get("country") or "").strip()
            tz_name = (profile.get("tz") or "").strip()
            lang = (profile.get("lang") or "en").lower()
            
            if not (city and country and tz_name):
                continue
            
            try:
                tz = zoneinfo.ZoneInfo(tz_name)
            except Exception:
                continue
            
            now_local = datetime.now(tz)
            
            # Fetch today's prayer times
            try:
                d = await aladhan_fetch(city, country, None)
                timings = d.get("timings", {}) or {}
            except Exception as e:
                print(f"[SCHED] Failed to fetch prayer times for {wa_id}: {e}")
                continue
            
            # Check each prayer time
            for prayer_name in PRAYER_ORDER:
                if prayer_name not in timings:
                    continue
                
                # Parse prayer time
                prayer_time_str = timings[prayer_name]
                try:
                    # Extract HH:MM from prayer time string
                    import re
                    time_match = re.search(r"(\d{1,2}):(\d{2})", prayer_time_str)
                    if not time_match:
                        continue
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2))
                    prayer_dt = datetime(now_local.year, now_local.month, now_local.day, hour, minute, tzinfo=tz)
                    
                    # If prayer time has already passed today, skip it
                    if prayer_dt < now_local:
                        continue
                except Exception:
                    continue
                
                # Calculate reminder time (10 minutes before)
                reminder_dt = prayer_dt - timedelta(minutes=10)
                
                # Skip if reminder time has already passed (shouldn't happen if prayer hasn't passed, but safety check)
                if reminder_dt < now_local:
                    continue
                
                # Check if we're within 1 minute of the reminder time
                time_diff = abs((now_local - reminder_dt).total_seconds())
                if time_diff > 60:  # More than 1 minute away
                    continue
                
                # Check deduplication - ensure we haven't sent this reminder today
                reminder_key = f"prayer_reminder:{wa_id}:{prayer_name}:{now_local.date().isoformat()}"
                if await r.get(reminder_key):
                    continue  # Already sent today
                
                # Build reminder message
                prayer_time_display = prayer_dt.strftime("%H:%M")
                if lang == "ar":
                    prayer_names_ar = {
                        "Fajr": "الفجر", "Dhuhr": "الظهر", "Asr": "العصر",
                        "Maghrib": "المغرب", "Isha": "العشاء"
                    }
                    prayer_ar = prayer_names_ar.get(prayer_name, prayer_name)
                    msg = f"⏰ تذكير: صلاة {prayer_ar} في الساعة {prayer_time_display} (خلال 10 دقائق)"
                else:
                    msg = f"⏰ Reminder: {prayer_name} prayer at {prayer_time_display} (in 10 minutes)"
                
                # Send reminder
                try:
                    await send_text(wa_id, msg)
                    # Mark as sent (expires after 24 hours)
                    await r.set(reminder_key, "1", ex=24 * 3600)
                    print(f"[SCHED] Sent {prayer_name} reminder to {wa_id} at {now_local.strftime('%H:%M')}")
                except Exception as e:
                    print(f"[SCHED] Failed to send prayer reminder to {wa_id}: {e}")
        
        except Exception as e:
            print(f"[SCHED] Prayer reminder tick failed for {wa_id}: {e}")
