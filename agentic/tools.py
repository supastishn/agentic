import os
import subprocess
import json
from pathlib import Path
import requests
from rich.console import Console
from rich.panel import Panel

# --- Tool Implementations ---

def read_file(path: str, read_files_in_session: set) -> str:
    """Reads the content of a single file."""
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"

    if path in read_files_in_session:
        return f"File '{path}' has already been read in this session. Its contents are in the context."

    try:
        content = p.read_text()
        read_files_in_session.add(path)
        return content
    except Exception as e:
        return f"Error reading file {path}: {e}"

def read_many_files(paths: list[str], read_files_in_session: set) -> str:
    """Reads and returns the content of multiple files."""
    results = []
    for path in paths:
        content = read_file(path, read_files_in_session)
        results.append({"path": path, "content": content})
    return json.dumps(results)

def read_folder(path: str = ".") -> str:
    """Lists files and directories at a given path."""
    p = Path(path)
    if not p.is_dir():
        return f"Error: Path '{path}' is not a valid directory."
    try:
        items = []
        for item in sorted(p.iterdir()):
            if item.is_dir():
                items.append(f"{item.name}/")
            else:
                items.append(item.name)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing files at {path}: {e}"

def find_files(pattern: str) -> str:
    """Finds files matching a glob pattern recursively."""
    try:
        files = [str(p) for p in Path.cwd().rglob(pattern)]
        if not files:
            return f"No files found matching pattern: {pattern}"
        return "\n".join(files)
    except Exception as e:
        return f"Error finding files: {e}"

def search_text(query: str, file_path: str) -> str:
    """Searches for a query string in a file and returns matching lines with line numbers."""
    p = Path(file_path)
    if not p.is_file():
        return f"Error: File not found at {file_path}"
    try:
        matching_lines = []
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if query in line:
                    matching_lines.append(f"{i}:{line.strip()}")
        if not matching_lines:
            return f"No matches for '{query}' found in {file_path}."
        return "\n".join(matching_lines)
    except Exception as e:
        return f"Error searching file {file_path}: {e}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file, overwriting it if it exists or creating it if it doesn't."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing to file {path}: {e}"

def edit(path: str, search: str, replace: str) -> str:
    """Replaces the first occurrence of a search string in a file with a replace string."""
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"
    try:
        original_content = p.read_text()
        if search not in original_content:
            return "ERROR: Search string not found in the file."
        new_content = original_content.replace(search, replace, 1)
        p.write_text(new_content)
        return f"Successfully edited {path}."
    except Exception as e:
        return f"Error editing file {path}: {e}"

def shell(command: str) -> str:
    """Runs a shell command and returns its output (stdout and stderr)."""
    try:
        result = subprocess.run(
            command, shell=True, text=True, capture_output=True, check=False,
        )
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error running command '{command}': {e}"

def web_fetch(url: str) -> str:
    """Fetches the text content of a web page."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"

def think(thought: str) -> str:
    """
    Processes a thought by thinking about it deeply, considering related info, code, etc.
    This helps in breaking down complex problems and forming a plan.
    """
    console = Console()
    console.print(
        Panel(
            thought,
            title="[bold magenta]Thinking[/]",
            border_style="magenta",
            expand=False
        )
    )
    return "Thought successfully processed!"

def save_memory(text: str) -> str:
    """Use to remember a key piece of information by adding it to the conversation context."""
    return f"OK, I will remember this: '{text}'"

# --- Tool Definitions for the LLM ---

AVAILABLE_TOOLS = {
    "Edit": edit,
    "FindFiles": find_files,
    "ReadFile": read_file,
    "ReadFolder": read_folder,
    "ReadManyFiles": read_many_files,
    "SaveMemory": save_memory,
    "SearchText": search_text,
    "Shell": shell,
    "Think": think,
    "WebFetch": web_fetch,
    "WriteFile": write_file,
}

TOOLS_METADATA = [
    {"type": "function", "function": {"name": "ReadFolder", "description": "Lists files and directories in a specified path. Use '.' for the current directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the directory."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "FindFiles", "description": "Finds files recursively using a glob pattern (e.g., '**/*.py').", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The glob pattern to search for."}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "ReadFile", "description": "Reads the entire content of a single file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "ReadManyFiles", "description": "Reads the contents of multiple files at once.", "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}, "description": "A list of relative paths to the files."}}, "required": ["paths"]}}},
    {"type": "function", "function": {"name": "SearchText", "description": "Searches for a text query within a single file and returns matching lines.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The text to search for."}, "file_path": {"type": "string", "description": "The path of the file to search in."}}, "required": ["query", "file_path"]}}},
    {"type": "function", "function": {"name": "WriteFile", "description": "Writes content to a file, creating it if it doesn't exist or overwriting it completely if it does.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file."}, "content": {"type": "string", "description": "The full content to write to the file."}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Performs a targeted search-and-replace on a file. Safer than WriteFile for small changes. Fails if the search string is not found.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file to edit."}, "search": {"type": "string", "description": "The exact text to find in the file."}, "replace": {"type": "string", "description": "The text to replace the 'search' text with."}}, "required": ["path", "search", "replace"]}}},
    {"type": "function", "function": {"name": "Shell", "description": "Executes a shell command and returns the output. Use with caution.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to execute."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Think", "description": "Processes a thought by thinking about it deeply, considering related info, code, etc. This helps in breaking down complex problems and forming a plan.", "parameters": {"type": "object", "properties": {"thought": {"type": "string", "description": "The thought to think about deeply. Think about related info, code, etc."}}, "required": ["thought"]}}},
    {"type": "function", "function": {"name": "WebFetch", "description": "Fetches the text content from a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to fetch content from."}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "SaveMemory", "description": "Use to remember a key piece of information. Adds the information to the conversation context.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The information to remember."}}, "required": ["text"]}}},
]
