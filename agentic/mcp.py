import requests
import json
from pathlib import Path
import os
import sys
from rich.console import Console

# --- Configuration File Management ---

def _get_config_paths() -> dict[str, Path]:
    """Returns the paths for user, project, and local MCP config files."""
    project_root = Path(os.getcwd())
    config_dir = Path.home() / ".agentic-cli-coder"
    
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

def save_mcp_config_for_scope(servers_data: dict, scope: str) -> str:
    """Saves a complete server dictionary to the specified scope's file."""
    paths = _get_config_paths()
    path = paths.get(scope)
    if not path:
        return f"Error: Invalid scope '{scope}'. Must be 'user', 'project', or 'local'."

    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # The data should be in the format {"servers": {...}}
        full_config = {"servers": servers_data}
        with path.open("w", encoding="utf-8") as f:
            json.dump(full_config, f, indent=2)
        
        return f"Successfully saved configuration to {scope} scope."
    except Exception as e:
        return f"Error saving configuration to {path}: {e}"

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

def copy_claude_code_mcp_config():
    """Finds and copies MCP server configs from Claude Code's desktop config."""
    console = Console()
    
    # 1. Find Claude Code config file path based on OS
    claude_config_path = None
    if sys.platform == "darwin": # macOS
        claude_config_path = Path.home() / ".claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        appdata = os.getenv("APPDATA")
        if appdata:
            claude_config_path = Path(appdata) / "Claude" / "claude_desktop_config.json"
    elif "linux" in sys.platform:
        claude_config_path = Path.home() / ".claude" / "claude_desktop_config.json"

    if not claude_config_path or not claude_config_path.exists():
        console.print("\n[bold red]Error:[/] Could not find the Claude Code configuration file.")
        console.input("Press Enter to continue...")
        return

    # 2. Read the Claude config file
    try:
        with claude_config_path.open("r", encoding="utf-8") as f:
            claude_config = json.load(f)
        claude_servers = claude_config.get("mcp_servers", {})
        if not claude_servers:
            console.print("\n[yellow]No MCP servers found in the Claude Code configuration.[/yellow]")
            console.input("Press Enter to continue...")
            return
    except (json.JSONDecodeError, IOError) as e:
        console.print(f"\n[bold red]Error reading Claude Code config file:[/] {e}")
        console.input("Press Enter to continue...")
        return

    # Correctly load just the user scope servers for comparison and update
    user_config_path = _get_config_paths()['user']
    agentic_user_servers = {}
    if user_config_path.exists():
        try:
            with user_config_path.open('r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    agentic_user_servers = json.loads(content).get('servers', {})
        except (json.JSONDecodeError, IOError):
            agentic_user_servers = {} # Start fresh if file is corrupt

    # 4. Merge configs (non-destructive)
    copied_servers = []
    skipped_servers = []
    
    for name, config_data in claude_servers.items():
        if name not in agentic_user_servers:
            agentic_user_servers[name] = config_data
            copied_servers.append(name)
        else:
            skipped_servers.append(name)

    # 5. Save the updated agentic user-scope config
    save_mcp_config_for_scope(agentic_user_servers, "user")

    # 6. Report results
    console.print("\n[bold green]✔ Import Complete[/bold green]")
    if copied_servers:
        console.print("Copied servers: " + ", ".join(f"[cyan]{s}[/]" for s in copied_servers))
    if skipped_servers:
        console.print("Skipped (already exist): " + ", ".join(f"[yellow]{s}[/]" for s in skipped_servers))
    if not copied_servers and not skipped_servers:
        console.print("No new servers to import.")

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
