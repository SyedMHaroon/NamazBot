"""
Utility script to subscribe/unsubscribe users to daily digest.
Usage:
    python subscribe_digest.py <wa_id>           # Subscribe
    python subscribe_digest.py <wa_id> --unsub   # Unsubscribe
    python subscribe_digest.py --list            # List all subscribers
"""
import asyncio
import sys
from digest_job import subscribe_to_digest, unsubscribe_from_digest, is_subscribed_to_digest
from data.redis_store import get_redis

async def main():
    if len(sys.argv) < 2:
        print("Usage: python subscribe_digest.py <wa_id> [--unsub|--list]")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        r = await get_redis()
        subs = await r.smembers("digest:subs")
        print(f"Total subscribers: {len(subs)}")
        for wa_id in subs:
            status = await is_subscribed_to_digest(wa_id)
            print(f"  - {wa_id}: {'✓' if status else '✗'}")
        return
    
    wa_id = sys.argv[1]
    is_unsub = "--unsub" in sys.argv or "-u" in sys.argv
    
    if is_unsub:
        success = await unsubscribe_from_digest(wa_id)
        print(f"Unsubscribed {wa_id}: {'Success' if success else 'Failed'}")
    else:
        success = await subscribe_to_digest(wa_id)
        print(f"Subscribed {wa_id}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())

