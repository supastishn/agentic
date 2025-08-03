import requests
import json
from pathlib import Path
import os

# --- Configuration File Management ---

def _get_config_paths() -> dict[str, Path]:
    """Returns the paths for user, project, and local MCP config files."""
    project_root = Path(os.getcwd())
    config_dir = Path.home() / ".agentic-pypi"
    
    # Sanitize project name for use in a file path
    safe_project_name = "".join(c for c in project_root.name if c.isalnum() or c in ('_', '-')).rstrip()

    return {
        "user": config_dir / "mcp.json",
        "project": project_root / ".agentic.mcp.json",
        "local": config_dir / "data" / "mcp" / f"{safe_project_name}.mcp.json",
    }

def load_mcp_servers() -> dict:
    """
    Loads MCP server configurations from all scopes (user, project, local)
    and merges them. Local settings override project, which override user.
    """
    paths = _get_config_paths()
    merged_servers = {}

    for scope in ["user", "project", "local"]:
        path = paths[scope]
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    merged_servers.update(data.get("servers", {}))
            except (json.JSONDecodeError, IOError):
                # Ignore corrupted or unreadable files
                pass
    return merged_servers

def save_mcp_server(name: str, config: dict, scope: str) -> str:
    """Saves or updates a server configuration in the specified scope's file."""
    paths = _get_config_paths()
    path = paths.get(scope)
    if not path:
        return f"Error: Invalid scope '{scope}'. Must be 'user', 'project', or 'local'."

    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        data = {"servers": {}}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                # Handle empty or invalid JSON file
                try:
                    data = json.load(f)
                    if "servers" not in data:
                        data["servers"] = {}
                except json.JSONDecodeError:
                    pass

        data["servers"][name] = config
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        return f"Successfully saved server '{name}' to {scope} scope."
    except Exception as e:
        return f"Error saving configuration to {path}: {e}"

def remove_mcp_server(name: str, scope: str) -> str:
    """Removes a server configuration from the specified scope's file."""
    paths = _get_config_paths()
    path = paths.get(scope)
    if not path:
        return f"Error: Invalid scope '{scope}'. Must be 'user', 'project', or 'local'."

    if not path.exists():
        return f"Error: No configuration file found for scope '{scope}'."

    try:
        data = {"servers": {}}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if name in data.get("servers", {}):
            del data["servers"][name]
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return f"Successfully removed server '{name}' from {scope} scope."
        else:
            return f"Error: Server '{name}' not found in {scope} scope."
    except Exception as e:
        return f"Error updating configuration file {path}: {e}"

# --- MCP Server Interaction ---

def list_server_tools(server_name: str) -> str:
    """Fetches the list of available tools from a configured MCP server."""
    servers = load_mcp_servers()
    server_config = servers.get(server_name)
    if not server_config:
        return f"Error: MCP server '{server_name}' not found in any configuration."

    url = server_config.get("url")
    if not url:
        return f"Error: Server '{server_name}' configuration is missing a 'url'."

    try:
        response = requests.get(f"{url}/tools", headers=server_config.get("headers"), timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error connecting to MCP server at {url}: {e}"

def run_server_tool(server_name: str, tool_name: str, parameters: dict) -> str:
    """Executes a specific tool on a configured MCP server."""
    servers = load_mcp_servers()
    server_config = servers.get(server_name)
    if not server_config:
        return f"Error: MCP server '{server_name}' not found in any configuration."

    url = server_config.get("url")
    if not url:
        return f"Error: Server '{server_name}' configuration is missing a 'url'."

    try:
        payload = {"tool_name": tool_name, "parameters": parameters}
        response = requests.post(f"{url}/run_tool", json=payload, headers=server_config.get("headers"), timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error communicating with MCP server at {url}: {e}"
