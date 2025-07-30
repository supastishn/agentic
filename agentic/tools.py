import os
import subprocess
import json
from pathlib import Path

# --- Tool Implementations ---

def read_file(path: str, read_files_in_session: set) -> str:
    """
    Reads the content of a single file at the given path.
    If the file has already been read in this session, it returns a notification instead of the content.
    """
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
    """
    Reads and returns the content of multiple files.
    For each file, if it has already been read, returns a notification.
    """
    results = []
    for path in paths:
        content = read_file(path, read_files_in_session)
        results.append({"path": path, "content": content})
    return json.dumps(results)

def write_file(path: str, content: str) -> str:
    """
    Writes content to a file, overwriting it if it exists. Creates parent directories if needed.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing to file {path}: {e}"

def run_command(command: str) -> str:
    """
    Runs a shell command and returns its output (stdout and stderr).
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error running command '{command}': {e}"

def list_files(path: str = ".") -> str:
    """
    Lists files and directories at a given path.
    """
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

def create_file(path: str, content: str = "") -> str:
    """
    Creates a new file with the specified content. Fails if the file already exists.
    """
    p = Path(path)
    if p.exists():
        return f"Error: File '{path}' already exists. Use write_file to overwrite."

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully created file: {path}"
    except Exception as e:
        return f"Error creating file {path}: {e}"

# --- Tool Definitions for the LLM ---

AVAILABLE_TOOLS = {
    "read_file": read_file,
    "read_many_files": read_many_files,
    "write_file": write_file,
    "create_file": create_file,
    "list_files": list_files,
    "run_command": run_command,
}

TOOLS_METADATA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists files and directories in a specified path. Use '.' for the current directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the directory."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the entire content of a single file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_many_files",
            "description": "Reads the contents of multiple files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of relative paths to the files.",
                    },
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Creates a new file with specified content. It fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path for the new file."},
                    "content": {"type": "string", "description": "The initial content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes or overwrites a file with new content. To edit a file, first read it, then write the full modified content. Creates parent directories if they don't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file."},
                    "content": {"type": "string", "description": "The new, full content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a shell command and returns the output. Use with caution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute."},
                },
                "required": ["command"],
            },
        },
    },
]
