# mcp_client.py
import json
from typing import Optional, Dict, Any, List
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from data.db_postgres import get_zapier_mcp_url

# Cache for active clients (wa_id -> Client)
_client_cache: Dict[str, Client] = {}


async def get_mcp_client(wa_id: str) -> Optional[Client]:
    """Get or create MCP client for a user."""
    if wa_id in _client_cache:
        return _client_cache[wa_id]
    
    mcp_url = await get_zapier_mcp_url(wa_id)
    if not mcp_url:
        return None
    
    try:
        transport = StreamableHttpTransport(mcp_url)
        client = Client(transport=transport)
        _client_cache[wa_id] = client
        return client
    except Exception as e:
        print(f"[MCP] Failed to create client for {wa_id}: {e}")
        return None


async def is_calendar_connected(wa_id: str) -> bool:
    """Check if user has connected their calendar."""
    mcp_url = await get_zapier_mcp_url(wa_id)
    return mcp_url is not None


async def call_calendar_tool(wa_id: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a calendar tool via MCP.
    Returns {"success": bool, "data": Any, "error": Optional[str]}
    """
    client = await get_mcp_client(wa_id)
    if not client:
        return {
            "success": False,
            "data": None,
            "error": "Calendar not connected. Please connect your calendar first."
        }
    
    try:
        async with client:
            if not client.is_connected():
                return {
                    "success": False,
                    "data": None,
                    "error": "Failed to connect to calendar service."
                }
            
            result = await client.call_tool(tool_name, params)
            
            # Parse JSON result
            if result.content and len(result.content) > 0:
                content_text = result.content[0].text
                try:
                    json_result = json.loads(content_text)
                    return {"success": True, "data": json_result, "error": None}
                except json.JSONDecodeError:
                    return {"success": True, "data": content_text, "error": None}
            
            return {"success": True, "data": None, "error": None}
            
    except Exception as e:
        error_msg = str(e)
        print(f"[MCP] Tool call failed for {wa_id}: {error_msg}")
        return {
            "success": False,
            "data": None,
            "error": f"Calendar operation failed: {error_msg}"
        }


async def list_calendar_tools(wa_id: str) -> List[str]:
    """List available calendar tools for a user."""
    client = await get_mcp_client(wa_id)
    if not client:
        return []
    
    try:
        async with client:
            tools = await client.list_tools()
            return [t.name for t in tools]
    except Exception:
        return []

