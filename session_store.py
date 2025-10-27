# session_store.py
from typing import Dict, Any

# Simple in-memory store: phone -> profile dict
# Swap with Redis/DB later if you need persistence or scaling
WA_SESSIONS: Dict[str, Dict[str, Any]] = {}

def get_profile(phone: str) -> Dict[str, Any]:
    return WA_SESSIONS.get(phone, {})

def set_profile(phone: str, profile: Dict[str, Any]) -> None:
    WA_SESSIONS[phone] = profile
