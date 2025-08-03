import requests
import json

def list_server_tools(server_url: str) -> str:
    """
    Fetches the list of available tools from an MCP server.
    Makes a GET request to the server's /tools endpoint.
    """
    try:
        response = requests.get(f"{server_url}/tools", timeout=10)
        response.raise_for_status()
        # The server is expected to return a JSON list of tool definitions
        return response.text # Return raw text to be parsed by LLM or tool
    except requests.RequestException as e:
        return f"Error connecting to MCP server at {server_url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def run_server_tool(server_url: str, tool_name: str, parameters: dict) -> str:
    """
    Executes a specific tool on an MCP server.
    Makes a POST request to the server's /run_tool endpoint.
    """
    try:
        payload = {
            "tool_name": tool_name,
            "parameters": parameters
        }
        response = requests.post(f"{server_url}/run_tool", json=payload, timeout=30)
        response.raise_for_status()
        # The server's response can be JSON or plain text
        return response.text
    except requests.RequestException as e:
        return f"Error communicating with MCP server at {server_url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
