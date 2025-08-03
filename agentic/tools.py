import os
import subprocess
import json
import glob
from pathlib import Path
import requests
from rich.console import Console
from rich.panel import Panel
import ast

from . import browser
from . import config
from . import mcp

def generate_xml_tool_prompt(tools_metadata: list) -> str:
    """Generates a prompt explaining how to use tools with XML syntax."""
    if not tools_metadata:
        return ""

    prompt_parts = [
        "You have access to a set of tools to answer user questions. When you need to use a tool, you must respond with a `<tool_code>` block containing the XML for the tool call. The tool name is the main tag, and parameters are nested tags.",
        "For example, to call the 'ReadFile' tool with path 'src/main.py', you would respond with:",
        "<tool_code>\n<ReadFile>\n    <path>src/main.py</path>\n</ReadFile>\n</tool_code>",
        "\nIf you need to call multiple tools, provide multiple `<tool_code>` blocks.",
        "\nHere are the available tools:",
        "<tools>"
    ]

    for tool in tools_metadata:
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        
        tool_str = f'  <tool name="{name}">\n'
        tool_str += f'    <description>{description}</description>\n'
        
        params = func.get("parameters", {}).get("properties", {})
        if params:
            tool_str += '    <parameters>\n'
            required_params = func.get("parameters", {}).get("required", [])
            for param_name, param_info in params.items():
                param_type = param_info.get("type")
                is_required = "true" if param_name in required_params else "false"
                tool_str += f'      <param name="{param_name}" type="{param_info.get("type")}" required="{is_required}" />\n'
            tool_str += '    </parameters>\n'
        
        tool_str += '  </tool>'
        prompt_parts.append(tool_str)
        
    prompt_parts.append("</tools>")
    return "\n".join(prompt_parts)

# --- Tool Implementations ---

def browser_start() -> str:
    """Starts the headless browser. Must be called before any other browser action."""
    return browser.browser_manager.start()

def browser_close() -> str:
    """Closes the headless browser."""
    return browser.browser_manager.close()

def browser_navigate(url: str) -> str:
    """Navigates the current browser page to a URL."""
    return browser.browser_manager.navigate(url)

def browser_get_content() -> str:
    """Returns the full HTML content of the current browser page."""
    return browser.browser_manager.get_content()

def browser_click(selector: str) -> str:
    """Clicks on an element specified by a CSS selector."""
    return browser.browser_manager.click(selector)

def browser_type_text(selector: str, text: str) -> str:
    """Types text into an input field specified by a CSS selector."""
    return browser.browser_manager.type_text(selector, text)

def read_file(path: str, read_files_in_session: set, messages: list = None) -> str:
    """Reads the content of a single file."""
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"

    if path in read_files_in_session:
        # Remove previous instances of this file from context and replace with redaction message
        if messages:
            file_identifier = f"File: {path}"
            redaction_message = f"REDACTED TO SAVE TOKENS. PLEASE USE LATEST READ OF THE FILE: {path}"
            
            # Iterate through messages to find and redact previous file reads
            for msg in messages:
                if msg["role"] == "user" and file_identifier in msg["content"]:
                    # Replace the content with redaction message
                    msg["content"] = redaction_message
        
        return f"File '{path}' has already been read in this session. Its contents are in the context."

    try:
        content = p.read_text()
        read_files_in_session.add(path)
        return content
    except Exception as e:
        return f"Error reading file {path}: {e}"

def read_many_files(paths: list[str], read_files_in_session: set, messages: list = None) -> str:
    """Reads and returns the content of multiple files."""
    results = []
    for path in paths:
        content = read_file(path, read_files_in_session, messages)
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
        # Use glob.glob for robust cross-platform support of absolute/relative paths.
        files = glob.glob(pattern, recursive=True)
        if not files:
            return f"No files found matching pattern: {pattern}"
        return "\n".join(sorted(files)) # Sorting for consistent output
    except Exception as e:
        return f"Error finding files: {e}"

def list_symbols(path: str) -> str:
    """Lists functions and classes in a Python file using AST parsing."""
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"
    if not path.endswith('.py'):
        return "Error: ListSymbols currently only supports Python (.py) files."

    try:
        content = p.read_text()
        tree = ast.parse(content, filename=path)
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
                symbols.append(f"{symbol_type}: {node.name} (line {node.lineno})")
        return "\n".join(sorted(symbols)) if symbols else "No symbols found."
    except Exception as e:
        return f"Error parsing symbols in {path}: {e}"

def read_symbol(path: str, symbol_name: str, read_symbols_in_session: dict = None) -> str:
    """Reads the source code of a specific function or class from a Python file."""
    p = Path(path)
    if not p.is_file():
        return f"Error: File not found at {path}"
    if not path.endswith('.py'):
        return "Error: ReadSymbol currently only supports Python (.py) files."

    # Create identifier for this symbol
    symbol_identifier = f"{path}:{symbol_name}"
    
    if read_symbols_in_session is not None and symbol_identifier in read_symbols_in_session:
        return f"Symbol '{symbol_name}' from '{path}' has already been read in this session. Its contents are in the context."

    try:
        source_code = p.read_text()
        tree = ast.parse(source_code, filename=path)

        target_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if node.name == symbol_name:
                    target_node = node
                    break

        if not target_node:
            return f"Error: Symbol '{symbol_name}' not found in {path}."

        result = ast.get_source_segment(source_code, target_node)
        
        # Track this symbol read if tracking is enabled
        if read_symbols_in_session is not None:
            read_symbols_in_session[symbol_identifier] = True
            
        return result
    except Exception as e:
        return f"Error reading symbol '{symbol_name}' from {path}: {e}"

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

def git(command: str) -> str:
    """
    Runs a git command. Only a subset of commands are allowed.
    Allowed commands: rm, add, commit, diff, log.
    Example: Git(command="commit -m 'Initial commit'")
    """
    command_parts = command.strip().split()
    subcommand = command_parts[0] if command_parts else ""

    allowed_subcommands = {"rm", "add", "commit", "diff", "log"}
    if subcommand not in allowed_subcommands:
        return f"Error: git subcommand '{subcommand}' is not allowed. Allowed are: {', '.join(allowed_subcommands)}"

    full_command = f"git {command}"
    
    try:
        result = subprocess.run(
            full_command, shell=True, text=True, capture_output=True, check=False,
        )
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        return output
    except Exception as e:
        return f"Error running command '{full_command}': {e}"

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
    except Exception as e:
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

def user_input(question: str) -> str:
    """
    Asks the user a question to get feedback, clarification, or the next task.
    Use this when you are unsure how to proceed or want to confirm a plan.
    """
    console = Console()
    console.print(
        Panel(
            question,
            title="[bold yellow]Input Required[/]",
            border_style="yellow",
            expand=False
        )
    )
    return console.input("[bold yellow]Your response: [/]")

def mcp_list_tools(server_name: str) -> str:
    """
    Lists the available tools on a registered MCP (Model Context Protocol) server.
    """
    # This now directly calls the refactored mcp function.
    return mcp.list_server_tools(server_name)

def mcp_run_tool(server_name: str, tool_name: str, parameters: str) -> str:
    """
    Runs a tool on a registered MCP (Model Context Protocol) server.
    """
    try:
        params_dict = json.loads(parameters)
    except json.JSONDecodeError:
        return "Error: The 'parameters' argument must be a valid JSON string. Example: '{\"key\": \"value\"}'"

    # This now directly calls the refactored mcp function.
    return mcp.run_server_tool(server_name, tool_name, params_dict)

def save_memory(text: str, scope: str = "project", source: str = "llm") -> str:
    """
    Saves a key piece of information to a memory file that will be loaded at the start of future sessions.
    Scope can be 'project' (default) or 'global'.
    Source can be 'llm' (default) or 'user', which helps distinguish memory origins.
    """
    try:
        config._ensure_data_dir()

        if scope == "project":
            project_name = os.path.basename(os.getcwd())
            memory_file = config.DATA_DIR / f"{project_name}.md"
        elif scope == "global":
            memory_file = config.DATA_DIR / "memorys.global.md"
        else:
            return "Error: Invalid scope. Must be 'project' or 'global'."

        source_comment = f"<!-- {source.upper()} Generated Memory -->"
        with memory_file.open("a", encoding="utf-8") as f:
            # Prepend separator if file is not empty.
            if os.path.getsize(memory_file) > 0:
                f.write("\n\n---\n\n")
            f.write(f"{source_comment}\n{text}")

        return f"OK, I will remember this for future '{scope}' sessions."
    except Exception as e:
        return f"Error saving memory: {e}"

def end_task(reason: str, info: str = "") -> str:
    """
    Signals that the sub-agent has completed its task, providing a reason and optional information.
    This is a placeholder; the actual implementation is in the `cli` module.
    This must be the final tool call from a sub-agent.
    """
    # This function is a placeholder. The actual logic is handled specially in `cli.py`.
    return "Sub-agent task completion is handled by the main application loop."

def make_subagent(mode: str, prompt: str) -> str:
    """
    Creates and runs a sub-agent with a specific mode and prompt to accomplish a task.
    This is a placeholder; the actual implementation is in the `cli` module which orchestrates agent execution.
    The sub-agent runs non-interactively. Forbidden modes: 'ask', 'agent-maker'.
    """
    # This function is a placeholder. The actual logic is handled specially in `cli.py`.
    return "Sub-agent execution is handled by the main application loop."

def make_todo_list(items: list[str]) -> str:
    """
    Creates a todo list with the given items.
    Returns a string representation of the todo list.
    """
    if not items:
        return "Error: Todo list cannot be empty."
    
    todo_list = {"items": items, "completed": []}
    import json
    return json.dumps(todo_list)

def check_todo_list(todo_list_json: str) -> str:
    """
    Checks the status of a todo list.
    Returns information about completed and pending items.
    """
    try:
        import json
        todo_list = json.loads(todo_list_json)
        items = todo_list.get("items", [])
        completed = todo_list.get("completed", [])
        
        pending = [item for item in items if item not in completed]
        
        result = {
            "total": len(items),
            "completed": len(completed),
            "pending": len(pending),
            "completed_items": completed,
            "pending_items": pending
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error parsing todo list: {e}"

def mark_todo_item_complete(todo_list_json: str, item: str) -> str:
    """
    Marks an item in the todo list as complete.
    Returns the updated todo list.
    """
    try:
        import json
        todo_list = json.loads(todo_list_json)
        items = todo_list.get("items", [])
        completed = todo_list.get("completed", [])
        
        if item not in items:
            return f"Error: Item '{item}' not found in todo list."
        
        if item not in completed:
            completed.append(item)
            todo_list["completed"] = completed
        
        return json.dumps(todo_list)
    except Exception as e:
        return f"Error updating todo list: {e}"

# --- Tool Definitions for the LLM ---

AVAILABLE_TOOLS = {
    "BrowserClick": browser_click,
    "BrowserClose": browser_close,
    "BrowserGetContent": browser_get_content,
    "BrowserNavigate": browser_navigate,
    "BrowserStart": browser_start,
    "BrowserTypeText": browser_type_text,
    "Edit": edit,
    "EndTask": end_task,
    "FindFiles": find_files,
    "Git": git,
    "ListSymbols": list_symbols,
    "MakeSubagent": make_subagent,
    "MakeTodoList": make_todo_list,
    "CheckTodoList": check_todo_list,
    "MarkTodoItemComplete": mark_todo_item_complete,
    "McpListTools": mcp_list_tools,
    "McpRunTool": mcp_run_tool,
    "ReadFile": read_file,
    "ReadFolder": read_folder,
    "ReadManyFiles": read_many_files,
    "ReadSymbol": read_symbol,
    "SaveMemory": save_memory,
    "SearchText": search_text,
    "Shell": shell,
    "Think": think,
    "UserInput": user_input,
    "WebFetch": web_fetch,
    "WriteFile": write_file,
}

TOOLS_METADATA = [
    {"type": "function", "function": {"name": "BrowserStart", "description": "Starts the headless browser. Must be called before any other browser operations.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "BrowserClose", "description": "Closes the headless browser, ending the current session.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "BrowserNavigate", "description": "Navigates the current browser page to a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to navigate to."}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "BrowserGetContent", "description": "Returns the full HTML content of the current page. Useful for understanding the page structure.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "BrowserClick", "description": "Clicks on an element specified by a CSS selector.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "description": "The CSS selector of the element to click (e.g., '#submit-button', '.link-class')."}}, "required": ["selector"]}}},
    {"type": "function", "function": {"name": "BrowserTypeText", "description": "Types text into an input field specified by a CSS selector.", "parameters": {"type": "object", "properties": {"selector": {"type": "string", "description": "The CSS selector of the input element."}, "text": {"type": "string", "description": "The text to type."}}, "required": ["selector", "text"]}}},
    {"type": "function", "function": {"name": "ReadFolder", "description": "Lists files and directories in a specified path. Use '.' for the current directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the directory."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "FindFiles", "description": "Finds files recursively using a glob pattern (e.g., '**/*.py').", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The glob pattern to search for."}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "ReadFile", "description": "Reads the entire content of a single file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "ReadManyFiles", "description": "Reads the contents of multiple files at once.", "parameters": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}, "description": "A list of relative paths to the files."}}, "required": ["paths"]}}},
    {"type": "function", "function": {"name": "ListSymbols", "description": "Lists all functions and classes in a Python (.py) file. Useful for quickly understanding a file's structure.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the Python file."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "ReadSymbol", "description": "Reads the full source code of a specific function or class from a Python (.py) file. Use 'ListSymbols' first to find the symbol name.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the Python file."}, "symbol_name": {"type": "string", "description": "The name of the function or class to read."}}, "required": ["path", "symbol_name"]}}},
    {"type": "function", "function": {"name": "SearchText", "description": "Searches for a text query within a single file and returns matching lines.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The text to search for."}, "file_path": {"type": "string", "description": "The path of the file to search in."}}, "required": ["query", "file_path"]}}},
    {"type": "function", "function": {"name": "WriteFile", "description": "Writes content to a file, creating it if it doesn't exist or overwriting it completely if it does.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file."}, "content": {"type": "string", "description": "The full content to write to the file."}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Performs a targeted search-and-replace on a file. Safer than WriteFile for small changes. Fails if the search string is not found.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "The relative path to the file to edit."}, "search": {"type": "string", "description": "The exact text to find in the file."}, "replace": {"type": "string", "description": "The text to replace the 'search' text with."}}, "required": ["path", "search", "replace"]}}},
    {"type": "function", "function": {"name": "Git", "description": "Executes a git command. Allowed subcommands: rm, add, commit, diff, log. Make regular commits. Use diffs to analyze changes.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The git command arguments (e.g., 'commit -m \\'Initial commit\\'' or 'diff')."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "EndTask", "description": "Signals the end of the sub-agent's task with a reason and optional info. This MUST be the final tool call made by a sub-agent.", "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "The reason for ending the task (e.g., 'success', 'failure', 'partial_success')."}, "info": {"type": "string", "description": "Optional detailed information about what was accomplished or what failed."}}, "required": ["reason"]}}},
    {"type": "function", "function": {"name": "MakeSubagent", "description": "Creates and runs a sub-agent to perform a specific task. The sub-agent runs non-interactively and returns a JSON string with 'reason' and 'info' fields detailing the outcome. Use this to delegate complex work. Forbidden modes: 'ask', 'agent-maker'.", "parameters": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["code", "architect"], "description": "The mode for the sub-agent to run in."}, "prompt": {"type": "string", "description": "The specific and detailed prompt for the sub-agent's task."}}, "required": ["mode", "prompt"]}}},
    {"type": "function", "function": {"name": "McpListTools", "description": "Lists the available tools on a registered MCP (Model Context Protocol) server. Use this to discover what actions an MCP server supports.", "parameters": {"type": "object", "properties": {"server_name": {"type": "string", "description": "The name of the MCP server as defined in the configuration."}}, "required": ["server_name"]}}},
    {"type": "function", "function": {"name": "McpRunTool", "description": "Runs a specific tool on a registered MCP server with the given parameters.", "parameters": {"type": "object", "properties": {"server_name": {"type": "string", "description": "The name of the MCP server."}, "tool_name": {"type": "string", "description": "The name of the tool to run, found via McpListTools."}, "parameters": {"type": "string", "description": "A JSON string representing a dictionary of parameters for the tool. Example: '{\"query\": \"latest news\"}'"}}, "required": ["server_name", "tool_name", "parameters"]}}},
    {"type": "function", "function": {"name": "MakeTodoList", "description": "Creates a todo list with the specified items. Returns a JSON representation of the todo list that can be used with other todo tools.", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}, "description": "List of todo items."}}, "required": ["items"]}}},
    {"type": "function", "function": {"name": "CheckTodoList", "description": "Checks the status of a todo list. Returns information about completed and pending items.", "parameters": {"type": "object", "properties": {"todo_list_json": {"type": "string", "description": "JSON representation of the todo list."}}, "required": ["todo_list_json"]}}},
    {"type": "function", "function": {"name": "MarkTodoItemComplete", "description": "Marks an item in the todo list as complete.", "parameters": {"type": "object", "properties": {"todo_list_json": {"type": "string", "description": "JSON representation of the todo list."}, "item": {"type": "string", "description": "The item to mark as complete."}}, "required": ["todo_list_json", "item"]}}},
    {"type": "function", "function": {"name": "Shell", "description": "Executes a shell command and returns the output. Use with caution.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command to execute."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Think", "description": "Processes a thought by thinking about it deeply, considering related info, code, etc. This helps in breaking down complex problems and forming a plan.", "parameters": {"type": "object", "properties": {"thought": {"type": "string", "description": "The thought to think about deeply. Think about related info, code, etc."}}, "required": ["thought"]}}},
    {"type": "function", "function": {"name": "UserInput", "description": "Asks the user a question to get feedback, clarification, or the next task. Use this when you are unsure how to proceed or want to confirm a plan.", "parameters": {"type": "object", "properties": {"question": {"type": "string", "description": "The question to ask the user."}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "WebFetch", "description": "Fetches the text content from a URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The URL to fetch content from."}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "SaveMemory", "description": "Saves a key piece of information to a persistent memory file (project-specific or global) to be loaded in future sessions. Use 'project' scope for context relevant only to the current directory, and 'global' for universally useful information.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The information to remember."}, "scope": {"type": "string", "enum": ["project", "global"], "description": "The scope of the memory, either 'project' or 'global'. Defaults to 'project'."}}, "required": ["text"]}}},
]
