# formatting.py
import re
from datetime import datetime

# ---------------- English & Arabic names (Gregorian) ----------------
EN_WEEKDAYS = {
    "Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday",
    "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"
}
EN_MONTHS = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December"
}

AR_WEEKDAYS = {
    "Mon": "الاثنين", "Tue": "الثلاثاء", "Wed": "الأربعاء", "Thu": "الخميس",
    "Fri": "الجمعة", "Sat": "السبت", "Sun": "الأحد"
}
AR_MONTHS = {
    "Jan": "يناير", "Feb": "فبراير", "Mar": "مارس", "Apr": "أبريل",
    "May": "مايو", "Jun": "يونيو", "Jul": "يوليو", "Aug": "أغسطس",
    "Sep": "سبتمبر", "Oct": "أكتوبر", "Nov": "نوفمبر", "Dec": "ديسمبر"
}

# Simple pattern: "Wed, 28 Oct 2025" or "28 Oct 2025"
DATE_PAT = re.compile(
    r"\b(?:(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s*)?(\d{1,2})\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b"
)

def _expand_en(m):
    wd, d, mon, y = m.groups()
    wd_full = f"{EN_WEEKDAYS.get(wd, wd)}, " if wd else ""
    return f"{wd_full}{int(d)} {EN_MONTHS[mon]} {y}"

def _expand_ar(m):
    wd, d, mon, y = m.groups()
    wd_full = f"{AR_WEEKDAYS.get(wd, wd)}، " if wd else ""
    return f"{wd_full}{int(d)} {AR_MONTHS[mon]} {y}"

def normalize_dates_in_text(text: str, lang: str) -> str:
    """Expand abbreviated dates like 'Wed, 28 Oct 2025' into full words."""
    if lang.lower().startswith("ar"):
        return DATE_PAT.sub(_expand_ar, text)
    return DATE_PAT.sub(_expand_en, text)

def normalize_prayer_names(text: str, lang: str) -> str:
    """Optional: expand prayer names so TTS pronounces clearly."""
    if lang.lower().startswith("ar"):
        text = re.sub(r"\bFajr\b", "الفجر", text)
        text = re.sub(r"\bDhuhr\b", "الظهر", text)
        text = re.sub(r"\bAsr\b", "العصر", text)
        text = re.sub(r"\bMaghrib\b", "المغرب", text)
        text = re.sub(r"\bIsha\b", "العشاء", text)
    else:
        text = re.sub(r"\bDhuhr\b", "Dhuhr (noon prayer)", text)
        text = re.sub(r"\bIsha\b", "Isha (night prayer)", text)
    return text

# ---------------- Hijri month names & helpers ----------------
HIJRI_EN = {
    1: "Muharram",
    2: "Safar",
    3: "Rabi al-Awwal",
    4: "Rabi al-Thani",
    5: "Jumada al-Awwal",
    6: "Jumada al-Thani",
    7: "Rajab",
    8: "Shaban",
    9: "Ramadan",
    10: "Shawwal",
    11: "Dhul Qadah",
    12: "Dhul Hijjah",
}
HIJRI_AR = {
    1: "محرم",
    2: "صفر",
    3: "ربيع الأول",
    4: "ربيع الآخر",
    5: "جمادى الأولى",
    6: "جمادى الآخرة",
    7: "رجب",
    8: "شعبان",
    9: "رمضان",
    10: "شوال",
    11: "ذو القعدة",
    12: "ذو الحجة",
}

def hijri_numeric_to_words(hijri_str: str, lang: str = "en") -> str:
    """
    Convert 'DD-MM-YYYY' (e.g., '05-05-1447') to '5 Jumada al-Ula 1447' (EN)
    or '5 جمادى الأولى 1447' (AR). Returns original if parsing fails.
    """
    m = re.fullmatch(r"\s*(\d{1,2})-(\d{1,2})-(\d{3,4})\s*", hijri_str)
    if not m:
        return hijri_str
    d, mm, yyyy = map(int, m.groups())
    names = HIJRI_AR if lang.lower().startswith("ar") else HIJRI_EN
    month_name = names.get(mm)
    if not month_name:
        return hijri_str
    return f"{d} {month_name} {yyyy}"

def replace_hijri_numbers_in_text(text: str, lang: str = "en") -> str:
    """
    Finds any 'DD-MM-YYYY' Hijri-looking tokens in text and replaces with words.
    """
    def _repl(m):
        return hijri_numeric_to_words(m.group(0), lang)
    return re.sub(r"\b\d{1,2}-\d{1,2}-\d{3,4}\b", _repl, text)

def format_hijri_from_aladhan(hijri: dict, lang: str = "en") -> str:
    """
    Build a full Hijri date string from Aladhan's structured fields.
    Expecting:
      hijri = { "day": "05", "month": {"en": "...", "ar": "..."}, "year": "1447", "weekday": {"en": "...", "ar": "..."} }
    """
    day = int(hijri.get("day", "1"))
    year = hijri.get("year", "")
    month = hijri.get("month", {})
    weekday = hijri.get("weekday", {})

    if lang.lower().startswith("ar"):
        wd = weekday.get("ar")
        mon = month.get("ar") or HIJRI_AR.get(int(month.get("number", 0)), "")
        if wd:
            return f"{wd}، {day} {mon} {year}"
        return f"{day} {mon} {year}"
    else:
        wd = weekday.get("en")
        mon = month.get("en") or HIJRI_EN.get(int(month.get("number", 0)), "")
        if wd:
            return f"{wd}, {day} {mon} {year} AH"
        return f"{day} {mon} {year} AH"

# ---------------- Main normalizer for TTS/text ----------------
def normalize_for_tts(text: str, lang: str) -> str:
    """
    Expand Gregorian abbreviations, normalize prayer names,
    and convert any numeric Hijri dates (DD-MM-YYYY) to words.
    """
    t = normalize_dates_in_text(text, lang)
    t = normalize_prayer_names(t, lang)
    t = replace_hijri_numbers_in_text(t, lang)
    return t
