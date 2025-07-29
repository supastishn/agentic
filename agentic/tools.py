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

def edit_file(path: str, search_string: str, replace_string: str) -> str:
    """
    Performs a search and replace on a file and writes the changes back.
    """
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"
    try:
        content = p.read_text()
        new_content = content.replace(search_string, replace_string)
        if content == new_content:
            return "No changes made: search string not found."
        p.write_text(new_content)
        return f"Successfully edited {path}."
    except Exception as e:
        return f"Error editing file {path}: {e}"

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


# --- Tool Definitions for the LLM ---

AVAILABLE_TOOLS = {
    "read_file": read_file,
    "read_many_files": read_many_files,
    "edit_file": edit_file,
    "run_command": run_command,
}

TOOLS_METADATA = [
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
            "name": "edit_file",
            "description": "Searches for a string in a file and replaces it. Overwrites the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file."},
                    "search_string": {"type": "string", "description": "The exact string to find."},
                    "replace_string": {"type": "string", "description": "The string to replace the search_string with."},
                },
                "required": ["path", "search_string", "replace_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a shell command and returns the output.",
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
