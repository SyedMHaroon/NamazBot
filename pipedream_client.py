# pipedream_client.py
import os
import json
import secrets
import httpx
from typing import Optional, Dict, Any, List
from data.db_postgres import get_pipedream_connection_id
from data.redis_store import cache_set, cache_get
from dotenv import load_dotenv

load_dotenv()

# Pipedream API configuration
PIPEDREAM_CLIENT_ID = os.getenv("PIPEDREAM_CLIENT_ID")
PIPEDREAM_CLIENT_SECRET = os.getenv("PIPEDREAM_CLIENT_SECRET")
PIPEDREAM_PROJECT_ID = os.getenv("PIPEDREAM_PROJECT_ID")
PIPEDREAM_ENVIRONMENT = (os.getenv("PIPEDREAM_ENVIRONMENT") or "development").strip().lower()
PIPEDREAM_API_KEY = os.getenv("PIPEDREAM_API_KEY")  # API key for Connect API (if different from OAuth)
PIPEDREAM_CREATE_EVENT_WEBHOOK = os.getenv("PIPEDREAM_CREATE_EVENT_WEBHOOK")
PIPEDREAM_API_BASE = "https://api.pipedream.com/v1"
PIPEDREAM_MCP_BASE = "https://remote.mcp.pipedream.net"
BASE_URL = os.getenv("BASE_URL")

# Log configuration on import (for debugging)
if PIPEDREAM_PROJECT_ID:
    print(f"[PIPEDREAM] Configuration loaded - Project ID: {PIPEDREAM_PROJECT_ID}, Environment: {PIPEDREAM_ENVIRONMENT}")
else:
    print(f"[PIPEDREAM] WARNING: PIPEDREAM_PROJECT_ID not set in environment")

# HTTP client for API calls
_http_client = httpx.AsyncClient(timeout=30.0)

# HTTP client for MCP calls with longer timeout (MCP can take longer)
_mcp_client = httpx.AsyncClient(timeout=120.0)  # 2 minutes for MCP calls

# Module cache for MCP initialization state per user
_mcp_initialized_for_user: Dict[str, bool] = {}


async def get_pipedream_access_token(scope: Optional[str] = None) -> Optional[str]:
    """
    Get Pipedream API access token using OAuth client credentials.
    
    Args:
        scope: Optional space-separated list of scopes. Defaults to "*" (full access).
               For Connect API, use "connect:tokens:create" or "connect:*"
    """
    if not PIPEDREAM_CLIENT_ID or not PIPEDREAM_CLIENT_SECRET:
        print("[PIPEDREAM] Missing CLIENT_ID or CLIENT_SECRET")
        return None
    
    try:
        # Request body - include scope if provided
        request_body = {
            "grant_type": "client_credentials",
            "client_id": PIPEDREAM_CLIENT_ID,
            "client_secret": PIPEDREAM_CLIENT_SECRET,
        }
        
        # Add scope if specified (defaults to "*" if not provided)
        # Note: For Connect API, usually no custom scope needed - default "*" works
        if scope:
            request_body["scope"] = scope
            print(f"[PIPEDREAM] Requesting access token with scope: {scope}")
        else:
            print("[PIPEDREAM] Requesting access token with default scope: * (no custom scope)")
        
        response = await _http_client.post(
            f"{PIPEDREAM_API_BASE}/oauth/token",
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()
        access_token = data.get("access_token")
        if access_token:
            print(f"[PIPEDREAM] Successfully obtained access token (length: {len(access_token)})")
        else:
            print(f"[PIPEDREAM] No access_token in response: {data}")
        return access_token
    except httpx.HTTPStatusError as e:
        print(f"[PIPEDREAM] Failed to get access token: HTTP {e.response.status_code}")
        print(f"[PIPEDREAM] Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"[PIPEDREAM] Failed to get access token: {e}")
        return None


async def _initialize_mcp(wa_id: str, access_token: str, account_id: Optional[str]) -> bool:
    """
    Initialize MCP server for a user (one-time per user session).
    Returns True if successful, False otherwise.
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",  # ✅ REQUIRED - fixes 406 error
        "x-pd-project-id": PIPEDREAM_PROJECT_ID,
        "x-pd-environment": PIPEDREAM_ENVIRONMENT,
        "x-pd-external-user-id": wa_id,
        "x-pd-tool-mode": "full-config",
        "x-pd-app-slug": "google_calendar",
    }
    
    if account_id and account_id.startswith("apn_"):
        headers["x-pd-account-id"] = account_id
    
    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "whatsapp-bot", "version": "1.0.0"},
        },
    }
    
    try:
        resp = await _mcp_client.post(PIPEDREAM_MCP_BASE, headers=headers, json=payload)
        body = await resp.aread()
        ct = resp.headers.get("content-type", "")
        
        if resp.status_code != 200:
            error_text = body.decode('utf-8', 'ignore') if body else f"HTTP {resp.status_code}"
            print(f"[PIPEDREAM] initialize HTTP {resp.status_code}: {error_text}")
            return False
        
        # Check if response is SSE or regular JSON
        if "event-stream" not in ct:
            # Regular JSON response
            if not body:
                print("[PIPEDREAM] initialize: empty response body")
                return False
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                print(f"[PIPEDREAM] initialize: failed to parse JSON: {e}, body: {body[:200]}")
                return False
        else:
            # SSE response - need to use stream
            print("[PIPEDREAM] initialize returned SSE, using stream...")
            async with _mcp_client.stream("POST", PIPEDREAM_MCP_BASE, headers=headers, json=payload) as sresp:
                sresp.raise_for_status()
                data = None
                async for line in sresp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            obj = json.loads(line[6:])
                            if "error" in obj or "result" in obj:
                                data = obj
                                break
                        except json.JSONDecodeError:
                            continue
                
                if data is None:
                    print("[PIPEDREAM] initialize: no data in SSE")
                    return False
        
        if "error" in data:
            print(f"[PIPEDREAM] initialize error: {data['error']}")
            return False
        
        # Check if we got a successful result
        if "result" not in data:
            print(f"[PIPEDREAM] initialize: no result in response, data: {data}")
            return False
        
        print(f"[PIPEDREAM] MCP initialize OK, result: {data.get('result')}")
        _mcp_initialized_for_user[wa_id] = True
        return True
    except Exception as e:
        print(f"[PIPEDREAM] initialize exception: {e}")
        import traceback
        print(f"[PIPEDREAM] Traceback: {traceback.format_exc()}")
        return False


async def trigger_workflow_create_event(wa_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trigger a Pipedream workflow (HTTP webhook) to create a calendar event.

    Returns dict with keys: success (bool), data (Any), error (Optional[str]), not_configured (bool)
    """
    if not PIPEDREAM_CREATE_EVENT_WEBHOOK:
        return {
            "success": False,
            "data": None,
            "error": "Workflow webhook not configured",
            "not_configured": True,
        }
    
    # Retrieve the user's account id so we can call the workflow with the same identity
    account_id = await get_pipedream_connection_id(wa_id)
    if not account_id or not account_id.startswith("apn_"):
        return {
            "success": False,
            "data": None,
            "error": "Calendar connection invalid or missing account id",
            "not_configured": False,
        }

    # Developer token needed for workflow
    access_token = await get_pipedream_access_token(scope=None)
    if not access_token:
        return {
            "success": False,
            "data": None,
            "error": "Failed to authenticate with Pipedream",
            "not_configured": False,
        }

    try:
        # Merge payload with metadata
        body = {
            "wa_id": wa_id,
            "source": "whatsapp-bot",
            **(payload or {}),
        }
        print(f"[PIPEDREAM] Triggering workflow webhook for wa_id={wa_id}")
        resp = await _http_client.post(
            PIPEDREAM_CREATE_EVENT_WEBHOOK,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {access_token}",
                "x-pd-project-id": PIPEDREAM_PROJECT_ID or "",
                "x-pd-environment": PIPEDREAM_ENVIRONMENT or "development",
                "x-pd-external-user-id": wa_id,
                "x-pd-account-id": account_id,
                "x-pd-tool-mode": "full-config",
                "x-pd-app-slug": "google_calendar",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            data = resp.text
        print(f"[PIPEDREAM] Workflow webhook success (status={resp.status_code})")
        return {
            "success": True,
            "data": data,
            "error": None,
            "not_configured": False,
        }
    except httpx.HTTPStatusError as e:
        error_text = e.response.text if e.response is not None else str(e)
        print(f"[PIPEDREAM] Workflow webhook HTTP error: {e.response.status_code} {error_text}")
        return {
            "success": False,
            "data": None,
            "error": f"HTTP {e.response.status_code}: {error_text}",
            "not_configured": False,
        }
    except Exception as e:
        import traceback
        print(f"[PIPEDREAM] Workflow webhook exception: {e}")
        print(f"[PIPEDREAM] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "data": None,
            "error": str(e) or repr(e),
            "not_configured": False,
        }


async def is_calendar_connected(wa_id: str) -> bool:
    """Check if user has connected their calendar."""
    connection_id = await get_pipedream_connection_id(wa_id)
    return connection_id is not None


async def get_oauth_authorization_url(wa_id: str, token: str) -> Optional[str]:
    """
    Create Connect token and get Pipedream OAuth authorization URL for Google Calendar.
    Returns the URL where users should be redirected to authorize.
    """
    # Get access token without custom scope (defaults to * which is full access)
    # Client credentials for Pipedream usually works without adding custom scopes
    access_token = await get_pipedream_access_token(scope=None)  # No scope = defaults to *
    if not access_token:
        print("[PIPEDREAM] No access token available for connect/tokens")
        return None
    
    if not PIPEDREAM_PROJECT_ID:
        print("[PIPEDREAM] Missing PIPEDREAM_PROJECT_ID - required for Connect token")
        return None
    
    try:
        # Create Connect token
        # Don't pass custom token in redirect URI - let Pipedream handle it
        success_redirect_uri = f"{BASE_URL}/auth/pipedream/callback?token={token}"
        error_redirect_uri = f"{BASE_URL}/auth/pipedream/callback?token={token}&error=1"
        
        print(f"[PIPEDREAM] Creating connect token for wa_id: {wa_id}, project_environment: {PIPEDREAM_ENVIRONMENT}")
        print(f"[PIPEDREAM] Project ID: {PIPEDREAM_PROJECT_ID}")
        print(f"[PIPEDREAM] Success redirect URI: {success_redirect_uri}")
        
        # Request body for Connect token - MUST include project_id and environment
        # Environment must be one of: "development" or "production"
        # Ensure environment is properly set
        env_value = PIPEDREAM_ENVIRONMENT.strip().lower() if PIPEDREAM_ENVIRONMENT else "development"
        if env_value not in ["development", "production"]:
            print(f"[PIPEDREAM] WARNING: Invalid environment '{env_value}', defaulting to 'development'")
            env_value = "development"
        
        # Try both "environment" and "project_environment" 
        # The SDK uses "project_environment" for client config, but API endpoint might use "environment"
        # Sending both to ensure API recognizes it
        request_body = {
            "external_user_id": wa_id,
            "project_id": PIPEDREAM_PROJECT_ID,  # Required in body, not just header
            "environment": env_value,  # API endpoint parameter name
            "project_environment": env_value,  # Also include (SDK uses this)
            "success_redirect_uri": success_redirect_uri,
            "error_redirect_uri": error_redirect_uri,
        }
        
        print(f"[PIPEDREAM] Request body: {request_body}")
        print(f"[PIPEDREAM] Environment value: '{env_value}' (type: {type(env_value)})")
        
        # Headers with access token (no custom scope needed)
        # API might require environment in header as well as body
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-pd-environment": env_value,  # Add environment as header (API requirement)
            "x-pd-project-id": PIPEDREAM_PROJECT_ID,  # Also include project_id in header
        }
        
        print(f"[PIPEDREAM] Calling Connect API with access token (Bearer)")
        print(f"[PIPEDREAM] Request headers: {headers}")
        # Try endpoint with project_id in path: /v1/connect/{project_id}/tokens
        response = await _http_client.post(
            f"{PIPEDREAM_API_BASE}/connect/{PIPEDREAM_PROJECT_ID}/tokens",
            headers=headers,
            json=request_body,
        )
        
        # Check if request was successful
        if response.status_code == 401:
            print(f"[PIPEDREAM] 401 Unauthorized")
            print(f"[PIPEDREAM] Response: {response.text}")
            print(f"[PIPEDREAM] Please verify:")
            print(f"[PIPEDREAM] 1. OAuth client belongs to same workspace as project {PIPEDREAM_PROJECT_ID}")
            print(f"[PIPEDREAM] 2. OAuth client has permission for Connect API")
            print(f"[PIPEDREAM] 3. Project ID {PIPEDREAM_PROJECT_ID} is correct")
        
        response.raise_for_status()
        data = response.json()
        print(f"[PIPEDREAM] Connect token created successfully: {data}")
        
        # Get connect_link_url from response
        # Pipedream returns connect_link_url or just the token
        connect_token = data.get("connect_link_url") or data.get("token") or data.get("id")
        
        if not connect_token:
            print(f"[PIPEDREAM] No connect_token in response: {data}")
            return None
        
        # If connect_token is already a full URL, append required parameters
        if connect_token.startswith("http"):
            # Pipedream's connect_link_url is basic, we need to add redirect_url and app
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            
            parsed = urlparse(connect_token)
            params = parse_qs(parsed.query)
            
            # Add required parameters for the authorization flow
            redirect_url = f"{BASE_URL}/auth/pipedream/callback"
            
            # Add redirect_url if not present (required for callback)
            if "redirect_url" not in params:
                params["redirect_url"] = [redirect_url]
            
            # Add app if not present (required for Google Calendar)
            if "app" not in params:
                params["app"] = ["google_calendar"]
            
            # Add external_user_id if not present
            if "external_user_id" not in params:
                params["external_user_id"] = [wa_id]
            
            # Reconstruct the URL with all parameters
            new_query = urlencode(params, doseq=True)
            auth_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment
            ))
        else:
            # Extract just the token if it's a URL
            if "token=" in str(connect_token):
                import re
                match = re.search(r'token=([^&]+)', str(connect_token))
                if match:
                    connect_token = match.group(1)
            
            # Construct URL with all required parameters (matching your format)
            redirect_url = f"{BASE_URL}/auth/pipedream/callback"
            
            auth_url = (
                f"https://pipedream.com/_static/connect.html"
                f"?token={connect_token}"
                f"&connectLink=true"
                f"&app=google_calendar"
                f"&external_user_id={wa_id}"
                f"&redirect_url={redirect_url}"
                f"&environment={PIPEDREAM_ENVIRONMENT}"
                f"&state={token}"  # Use our tracking token as state
            )
        
        print(f"[PIPEDREAM] Authorization URL: {auth_url}")
        return auth_url
    except httpx.HTTPStatusError as e:
        print(f"[PIPEDREAM] Failed to create connect token: HTTP {e.response.status_code}")
        print(f"[PIPEDREAM] Response: {e.response.text}")
        print(f"[PIPEDREAM] Request body: external_user_id={wa_id}, environment={PIPEDREAM_ENVIRONMENT}, project_id={PIPEDREAM_PROJECT_ID}")
        return None
    except Exception as e:
        print(f"[PIPEDREAM] Failed to create connect token: {e}")
        return None


async def exchange_code_for_connection_id(code: str, wa_id: str) -> Optional[str]:
    """
    After user authorizes, Pipedream redirects back.
    The connection is established, and we use wa_id as the connection identifier
    since we pass it as x-pd-external-user-id in MCP calls.
    """
    # For Pipedream Connect, we use the external_user_id (wa_id) as the connection identifier
    # since that's what we pass in x-pd-external-user-id header
    # The code parameter might contain connection info, but we use wa_id
    return wa_id


async def call_mcp_tool(wa_id: str, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a Google Calendar tool via Pipedream MCP using JSON-RPC.
    Returns {"success": bool, "data": Any, "error": Optional[str]}
    """
    # Get account_id (apn_*) from database - this is the connected account ID from Pipedream
    account_id = await get_pipedream_connection_id(wa_id)
    if not account_id:
        return {
            "success": False,
            "data": None,
            "error": "Calendar not connected. Please connect your calendar first."
        }
    
    # Check if account_id is valid (should start with "apn_")
    if not account_id.startswith("apn_"):
        print(f"[PIPEDREAM] WARNING: account_id '{account_id}' doesn't start with 'apn_'. This might cause MCP calls to fail.")
        print(f"[PIPEDREAM] Please re-authorize the calendar connection.")
        return {
            "success": False,
            "data": None,
            "error": "Calendar connection invalid. Please reconnect your calendar."
        }
    
    # Get developer access token (no scope needed for MCP calls)
    access_token = await get_pipedream_access_token(scope=None)
    if not access_token:
        return {
            "success": False,
            "data": None,
            "error": "Failed to authenticate with Pipedream."
        }
    
    # Ensure MCP is initialized for this user
    if not _mcp_initialized_for_user.get(wa_id):
        ok = await _initialize_mcp(wa_id, access_token, account_id)
        if not ok:
            return {"success": False, "data": None, "error": "Failed to initialize MCP session"}
    
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-pd-project-id": PIPEDREAM_PROJECT_ID,
            "x-pd-environment": PIPEDREAM_ENVIRONMENT,
            "x-pd-external-user-id": wa_id,
            "x-pd-tool-mode": "full-config",
            "x-pd-app-slug": "google_calendar",
        }
        
        # Add account_id if available (apn_*)
        if account_id and account_id.startswith("apn_"):
            headers["x-pd-account-id"] = account_id
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": params},
            "id": 1,
        }
        
        # Try non-stream POST first
        resp = await _mcp_client.post(PIPEDREAM_MCP_BASE, headers=headers, json=payload)
        body = await resp.aread()
        ct = resp.headers.get("content-type", "")
        
        if resp.status_code != 200:
            error_text = body.decode('utf-8', 'ignore') if body else f"HTTP {resp.status_code}"
            return {"success": False, "data": None, "error": f"HTTP {resp.status_code}: {error_text}"}
        
        # Check if response is SSE or regular JSON
        if "event-stream" not in ct:
            # Regular JSON response
            data = json.loads(body)
        else:
            # SSE response - need to use stream
            print("[PIPEDREAM] tools/call returned SSE, using stream...")
            async with _mcp_client.stream("POST", PIPEDREAM_MCP_BASE, headers=headers, json=payload) as sresp:
                sresp.raise_for_status()
                data = None
                async for line in sresp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            obj = json.loads(line[6:])
                            if "error" in obj or "result" in obj:
                                data = obj
                                break
                        except json.JSONDecodeError:
                            continue
                
                if data is None:
                    return {"success": False, "data": None, "error": "No data received from SSE"}
        
        if "error" in data:
            return {"success": False, "data": None, "error": data["error"].get("message", "MCP call failed")}
        
        result = data.get("result", {})
        if isinstance(result, dict) and "content" in result:
            content = result["content"]
            if isinstance(content, list) and content:
                text_content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                try:
                    return {"success": True, "data": json.loads(text_content), "error": None}
                except json.JSONDecodeError:
                    return {"success": True, "data": text_content, "error": None}
        
        return {"success": True, "data": result, "error": None}
        
    except httpx.HTTPStatusError as e:
        try:
            if hasattr(e.response, 'aread'):
                content = await e.response.aread()
                error_text = content.decode('utf-8', 'ignore') if content else f"HTTP {e.response.status_code}"
            else:
                error_text = e.response.text
            error_msg = f"HTTP {e.response.status_code}: {error_text}"
        except Exception:
            error_msg = f"HTTP {e.response.status_code}"
        print(f"[PIPEDREAM] MCP call failed: {error_msg}")
        return {
            "success": False,
            "data": None,
            "error": f"Calendar operation failed: {error_msg}"
        }
    except Exception as e:
        import traceback
        error_msg = str(e) or repr(e)
        error_type = type(e).__name__
        print(f"[PIPEDREAM] MCP call error: {error_type}: {error_msg}")
        print(f"[PIPEDREAM] Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "data": None,
            "error": f"Calendar operation failed: {error_type}: {error_msg}"
        }


async def list_mcp_tools(wa_id: str) -> List[str]:
    """List available MCP tools using JSON-RPC."""
    # Get account_id from database
    account_id = await get_pipedream_connection_id(wa_id)
    
    # Get developer access token (no scope needed for MCP calls)
    access_token = await get_pipedream_access_token(scope=None)
    if not access_token:
        return []
    
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "x-pd-project-id": PIPEDREAM_PROJECT_ID,
            "x-pd-environment": PIPEDREAM_ENVIRONMENT,
            "x-pd-external-user-id": wa_id,
            "x-pd-tool-mode": "full-config",
            "x-pd-app-slug": "google_calendar",
        }
        
        # Add account_id if available
        if account_id and account_id.startswith("apn_"):
            headers["x-pd-account-id"] = account_id
        
        # Initialize right before tools/list (server may be stateless)
        # Don't cache - initialize every time to ensure state is fresh
        print("[PIPEDREAM] Initializing MCP before tools/list...")
        init_ok = await _initialize_mcp(wa_id, access_token, account_id)
        if not init_ok:
            print("[PIPEDREAM] Failed to initialize, but continuing anyway...")
        
        payload = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
        
        # Use stream for tools/list (MCP server may require streaming for all calls)
        print("[PIPEDREAM] Calling tools/list with stream...")
        async with _mcp_client.stream("POST", PIPEDREAM_MCP_BASE, headers=headers, json=payload) as sresp:
            if sresp.status_code != 200:
                # Read error response
                try:
                    content = await sresp.aread()
                    error_text = content.decode('utf-8', 'ignore') if content else f"HTTP {sresp.status_code}"
                    print(f"[PIPEDREAM] tools/list HTTP {sresp.status_code}: {error_text}")
                except Exception as read_err:
                    print(f"[PIPEDREAM] tools/list HTTP {sresp.status_code}: Could not read error: {read_err}")
                return []
            
            sresp.raise_for_status()
            data = None
            async for line in sresp.aiter_lines():
                if line.startswith("data: "):
                    try:
                        obj = json.loads(line[6:])
                        if "error" in obj or "result" in obj:
                            data = obj
                            break
                    except json.JSONDecodeError:
                        continue
                elif line == "[DONE]":
                    break
            
            if data is None:
                print("[PIPEDREAM] tools/list: no data in SSE")
                return []
        
        if "error" in data:
            print(f"[PIPEDREAM] tools/list error: {data['error']}")
            return []
        
        tools = data.get("result", {}).get("tools", [])
        print(f"[PIPEDREAM] Tools discovered: {len(tools)}")
        for t in tools:
            print(f"[PIPEDREAM]   - {t.get('name')}")
        
        return [t["name"] for t in tools if "name" in t]
        
    except httpx.HTTPStatusError as e:
        print(f"[PIPEDREAM] Failed to list tools: HTTP {e.response.status_code}")
        try:
            if hasattr(e.response, 'aread'):
                content = await e.response.aread()
                error_text = content.decode('utf-8', 'ignore') if content else f"HTTP {e.response.status_code}"
            else:
                error_text = e.response.text
            print(f"[PIPEDREAM] Response: {error_text}")
        except Exception as read_err:
            print(f"[PIPEDREAM] Could not read response: {read_err}")
        return []
    except Exception as e:
        print(f"[PIPEDREAM] Failed to list tools: {e}")
        import traceback
        print(f"[PIPEDREAM] Traceback: {traceback.format_exc()}")
        return []


async def get_calendar_tool_name(wa_id: str, contains: str) -> Optional[str]:
    """
    Helper function to find a calendar tool name by searching for a keyword.
    Returns the first tool name that contains the keyword (case-insensitive).
    """
    names = await list_mcp_tools(wa_id)
    for n in names:
        if contains.lower() in n.lower():
            return n
    return None


async def create_calendar_event(wa_id: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a calendar event using Pipedream MCP."""
    # Dynamically find the create tool
    tool_name = await get_calendar_tool_name(wa_id, "create") or await get_calendar_tool_name(wa_id, "insert")
    if not tool_name:
        # Fallback to common tool names
        tool_name = "google_calendar_create_detailed_event"
    
    return await call_mcp_tool(wa_id, tool_name, event_data)


async def list_calendar_events(wa_id: str, start_time: str, end_time: str, calendar_id: str = "primary") -> Dict[str, Any]:
    """List calendar events using Pipedream MCP."""
    # Dynamically find the list/find tool
    tool_name = await get_calendar_tool_name(wa_id, "list") or await get_calendar_tool_name(wa_id, "find")
    if not tool_name:
        # Fallback to common tool names
        tool_name = "google_calendar_find_events"
    
    return await call_mcp_tool(
        wa_id,
        tool_name,
        {
            "calendarid": calendar_id,
            "start_time": start_time,
            "end_time": end_time,
            "ordering": "startTime",
        }
    )


async def search_calendar_events(wa_id: str, query: str, start_time: str, end_time: str) -> Dict[str, Any]:
    """Search calendar events using Pipedream MCP."""
    # Dynamically find the search/find tool
    tool_name = await get_calendar_tool_name(wa_id, "search") or await get_calendar_tool_name(wa_id, "find")
    if not tool_name:
        # Fallback to common tool names
        tool_name = "google_calendar_find_events"
    
    return await call_mcp_tool(
        wa_id,
        tool_name,
        {
            "q": query,
            "start_time": start_time,
            "end_time": end_time,
        }
    )


async def delete_calendar_event(wa_id: str, event_id: str, calendar_id: str = "primary") -> Dict[str, Any]:
    """Delete a calendar event using Pipedream MCP."""
    # Dynamically find the delete tool
    tool_name = await get_calendar_tool_name(wa_id, "delete") or await get_calendar_tool_name(wa_id, "remove")
    if not tool_name:
        # Fallback to common tool names
        tool_name = "google_calendar_delete_event"
    
    return await call_mcp_tool(
        wa_id,
        tool_name,
        {
            "eventid": event_id,
            "calendarid": calendar_id,
        }
    )


async def get_account_id_for_user(wa_id: str) -> Optional[str]:
    """
    Query Pipedream API to get the account_id (apn_*) for a user after authorization.
    Returns the latest healthy account_id found for the external_user_id.
    """
    access_token = await get_pipedream_access_token(scope=None)
    if not access_token:
        print("[PIPEDREAM] No access token for querying accounts")
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "x-pd-project-id": PIPEDREAM_PROJECT_ID,
            "x-pd-environment": PIPEDREAM_ENVIRONMENT,
        }
        
        url = f"{PIPEDREAM_API_BASE}/connect/{PIPEDREAM_PROJECT_ID}/accounts"
        params = {"external_user_id": wa_id}
        
        print(f"[PIPEDREAM] Querying accounts: {url} params={params}")
        resp = await _http_client.get(url, headers=headers, params=params)
        print(f"[PIPEDREAM] Accounts API status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        
        accounts = data.get("data") or data.get("accounts") or []
        if not accounts:
            print("[PIPEDREAM] No accounts found")
            return None
        
        # Pick healthy + latest updated_at
        def parse_ts(a):
            return a.get("updated_at") or a.get("created_at") or ""
        
        healthy = [a for a in accounts if a.get("healthy")]
        candidates = healthy or accounts
        candidates.sort(key=parse_ts, reverse=True)
        chosen = candidates[0]
        acc_id = chosen.get("id")
        print(f"[PIPEDREAM] Chosen account: {acc_id} (healthy={chosen.get('healthy')}, updated_at={chosen.get('updated_at')})")
        return acc_id
    except httpx.HTTPStatusError as e:
        print(f"[PIPEDREAM] Failed to query accounts: HTTP {e.response.status_code}")
        print(f"[PIPEDREAM] Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"[PIPEDREAM] get_account_id_for_user failed: {e}")
        import traceback
        print(f"[PIPEDREAM] Traceback: {traceback.format_exc()}")
        return None


# ---------- Authorization Token Management ----------
AUTH_TOKEN_TTL = 900  # 15 minutes in seconds

async def generate_auth_token(wa_id: str) -> str:
    """Generate unique auth token and store in Redis with wa_id mapping."""
    token = secrets.token_urlsafe(32)  # Generate secure random token
    key = f"auth_token:{token}"
    await cache_set(key, wa_id, AUTH_TOKEN_TTL)
    return token


async def get_wa_id_from_token(token: str) -> Optional[str]:
    """Get wa_id from auth token. Returns None if token is invalid or expired."""
    key = f"auth_token:{token}"
    wa_id = await cache_get(key)
    return wa_id if wa_id else None


async def invalidate_auth_token(token: str) -> None:
    """Invalidate an auth token by deleting it from Redis."""
    from data.redis_store import get_redis
    r = get_redis()
    key = f"auth_token:{token}"
    await r.delete(key)
