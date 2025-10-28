# test_readouts.py
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from voice_pipeline import tts_elevenlabs  # uses ELEVEN_API_KEY / ELEVEN_VOICE_ID

load_dotenv()
OUT = Path("tts_tests")
OUT.mkdir(exist_ok=True)

# --- Hardcoded lists ---

# 1) Counting 1-10
COUNT_EN = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
COUNT_AR = ["واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]

# 2) Hijri months (as we set in formatting.py)
HIJRI_EN = [
    "Muharram", "Safar", "Rabi al-Awwal", "Rabi al-Thani",
    "Jumada al-Ula", "Jumada al-Thaniyah", "Rajab", "Sha'ban",
    "Ramadan", "Shawwal", "Dhu al-Qa'dah", "Dhu al-Hijjah",
]
HIJRI_AR = [
    "محرم", "صفر", "ربيع الأول", "ربيع الآخر",
    "جمادى الأولى", "جمادى الآخرة", "رجب", "شعبان",
    "رمضان", "شوال", "ذو القعدة", "ذو الحجة",
]

# 3) Gregorian months
GREG_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
GREG_AR = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]

# --- Helpers ---

def join_for_readout(items, lang="en"):
    # adds a little natural phrasing
    if not items:
        return ""
    if lang.startswith("ar"):
        # simple Arabic join with “،” and “و”
        if len(items) == 1:
            return items[0]
        return "، ".join(items[:-1]) + "، و" + items[-1]
    else:
        # English Oxford comma style
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + ", and " + items[-1]

def script_for(title, items, lang="en"):
    if lang.startswith("ar"):
        return f"{title}:\n{join_for_readout(items, lang)}."
    else:
        return f"{title}: {join_for_readout(items, lang)}."

async def make_mp3(text: str, fname: str):
    audio = await tts_elevenlabs(text)
    path = OUT / fname
    if audio:
        path.write_bytes(audio)
        print(f"✅ Saved {path}  |  \"{text[:80]}{'...' if len(text)>80 else ''}\"")
    else:
        print(f"⚠️ TTS returned no audio for {fname}")

async def main():
    tasks = []

    # Counting
    tasks.append(make_mp3(script_for("Counting from one to ten", COUNT_EN, "en"), "count_en.mp3"))
    tasks.append(make_mp3(script_for("العد من واحد إلى عشرة", COUNT_AR, "ar"), "count_ar.mp3"))

    # Hijri months
    tasks.append(make_mp3(script_for("Hijri months in order", HIJRI_EN, "en"), "hijri_en.mp3"))
    tasks.append(make_mp3(script_for("أشهر التقويم الهجري بالترتيب", HIJRI_AR, "ar"), "hijri_ar.mp3"))

    # Gregorian months
    tasks.append(make_mp3(script_for("Gregorian months in order", GREG_EN, "en"), "greg_en.mp3"))
    tasks.append(make_mp3(script_for("أشهر التقويم الميلادي بالترتيب", GREG_AR, "ar"), "greg_ar.mp3"))

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
