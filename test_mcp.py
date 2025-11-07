# test_mcp.py
import asyncio
from pipedream_client import list_mcp_tools, call_mcp_tool, trigger_workflow_create_event

WA = "923267022223"

async def main():
    print("=" * 60)
    print("Testing Pipedream MCP Integration")
    print("=" * 60)
    
    # Test 1: List tools
    print("\n[TEST 1] Listing MCP tools...")
    tools = await list_mcp_tools(WA)
    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool}")
    
    if not tools:
        print("ERROR: No tools found via MCP! Continuing to workflow test...")
    
    # Test 2: Find create/insert tool
    print("\n[TEST 2] Finding create/insert tool...")
    create = next((t for t in tools if "insert" in t.lower() or "create" in t.lower()), None)
    print(f"Create tool: {create}")
    
    if not create:
        print("WARNING: No create/insert tool found via MCP. Workflow test will still run.")
    
    payload = {
        "summary": "MCP Test Event",
        "start": {"dateTime": "2025-11-06T23:00:00+05:00", "timeZone": "Asia/Karachi"},
        "end": {"dateTime": "2025-11-06T23:30:00+05:00", "timeZone": "Asia/Karachi"},
        "original_text": "schedule test event for 11pm to 11:30pm",
        "processed_text": "2025-11-06 schedule test event for 23:00 - 23:30",
    }

    print("\n[TEST 3A] Triggering workflow webhook (if configured)...")
    workflow_res = await trigger_workflow_create_event(WA, payload)
    print("Workflow result:", workflow_res)
    if workflow_res.get("success"):
        print("✅ Event created successfully via workflow webhook!")
    elif workflow_res.get("not_configured"):
        print("Workflow webhook not configured. Falling back to MCP call...")
        res = await call_mcp_tool(WA, create, payload)
        print("MCP result:", res)
        if res.get("success"):
            print("✅ Event created successfully via MCP!")
        else:
            print("❌ Failed to create event via MCP:", res.get("error"))
    else:
        print("❌ Workflow webhook call failed:", workflow_res.get("error"))
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())

