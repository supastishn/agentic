#!/usr/bin/env python3

import argparse
import sys
import json
import threading
import os
import re
import requests
import xml.etree.ElementTree as ET
import litellm
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import Frame
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.formatted_text import to_formatted_text

from . import tools
from . import config
from . import mcp # Add this import
from .rag import CodeRAG
from .tools import generate_xml_tool_prompt
from prompt_toolkit.filters import Filter
from prompt_toolkit.application.current import get_app

# Vendored filters from prompt-toolkit to avoid versioning issues.

class IsDone(Filter):
    """
    Filter that is `True` when the CLI is finished.
    """
    def __call__(self) -> bool:
        return get_app().is_done
    def __repr__(self) -> str:
        return "is_done"
is_done = IsDone()

class HasHistory(Filter):
    """
    Filter that is `True` if the current buffer has a history.
    """
    def __call__(self) -> bool:
        return len(get_app().current_buffer.history.get_strings()) > 0
    def __repr__(self) -> str:
        return "has_history"
has_history = HasHistory()


console = Console()

original_openai_api_base = os.environ.get("OPENAI_API_BASE")

def _update_environment_for_mode(agent_mode: str, cfg: dict):
    """Sets or unsets OPENAI_API_BASE based on the active mode's provider."""
    active_provider = None
    if cfg.get("temp_model"):
        active_provider = cfg["temp_model"].get("provider")
    else:
        modes = cfg.get("modes", {})
        global_config = modes.get("global", {})
        mode_config = modes.get(agent_mode, {})
        active_provider = mode_config.get("active_provider") or global_config.get("active_provider")

    if active_provider == "hackclub_ai":
        os.environ["OPENAI_API_BASE"] = "https://ai.hackclub.com"
    else:
        if original_openai_api_base is not None:
            os.environ["OPENAI_API_BASE"] = original_openai_api_base
        elif "OPENAI_API_BASE" in os.environ:
            del os.environ["OPENAI_API_BASE"]

def _update_session_stats(response, session_stats: dict, model_capabilities: dict):
    """Updates session token counts and costs from a litellm response object."""
    if not response or not response.usage:
        return

    current_prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    new_prompt_tokens = current_prompt_tokens - session_stats.get("last_prompt_tokens", 0)
    session_stats["last_prompt_tokens"] = current_prompt_tokens
    
    session_stats['prompt_tokens'] += new_prompt_tokens
    session_stats['completion_tokens'] += completion_tokens

    in_cost = model_capabilities.get("input_cost_per_token", 0) or 0
    out_cost = model_capabilities.get("output_cost_per_token", 0) or 0
    
    turn_cost = (new_prompt_tokens * in_cost) + (completion_tokens * out_cost)
    session_stats['cost'] += turn_cost

ASCII_LOGO = r"""
[bold green]                                         █████     ███          [/bold green]
[bold green]                                        ░░███     ░░░           [/bold green]
[bold cyan]  ██████    ███████  ██████  ████████   ███████   ████   ██████ [/bold cyan]
[bold cyan] ░░░░░███  ███░░███ ███░░███░░███░░███ ░░░███░   ░░███  ███░░███[/bold cyan]
[bold blue]  ███████ ░███ ░███░███████  ░███ ░███   ░███     ░███ ░███ ░░░ [/bold blue]
[bold blue] ███░░███ ░███ ░███░███░░░   ░███ ░███   ░███ ███ ░███ ░███  ███[/bold blue]
[bold magenta]░░████████░░███████░░██████  ████ █████  ░░█████  █████░░██████ [/bold magenta]
[bold magenta] ░░░░░░░░  ░░░░░███ ░░░░░░  ░░░░ ░░░░░    ░░░░░  ░░░░░  ░░░░░░  [/bold magenta]
[bold white]           ███ ░███                                             [/bold white]
[bold white]          ░░██████                                              [/bold white]
[bold white]           ░░░░░░                                               [/bold white]
"""

class SubAgentEndTask(Exception):
    def __init__(self, reason: str, info: str = ""):
        self.reason = reason
        self.info = info or "" # Ensure info is a string
        super().__init__(f"Sub-agent ended task with reason: {reason}")

CODE_SYSTEM_PROMPT = (
    "You are an AI assistant expert in software development. You have access to a powerful set of tools.\n"
    "Your primary directive is to ALWAYS understand the project context before providing code or solutions.\n\n"
    "**Mandatory Workflow:**\n"
    "1. **Consult Memories:** Check your PERMANENT MEMORIES first to see if you already have the context you need.\n"
    "2. **Gather Context:** If memories are insufficient, use `ReadFolder` to see the project layout. Then, use `ReadFile` on the most relevant files to understand how the code works.\n"
    "3. **Think & Plan:** Use the `Think` tool to break down the problem, formulate a hypothesis, and create a step-by-step plan. This is a crucial step for complex tasks.\n"
    "4. **Ask for Feedback (if needed):** If the plan is complex or you are unsure about the best approach, use the `UserInput` tool to ask for clarification or confirmation before proceeding.\n"
    "5. **Analyze & Execute:** Based on your plan, use `SearchText`, `Edit`, `WriteFile`, or `Git` to execute the steps. Make commits often.\n"
    "6. **Verify Changes:** After making changes, use `Git` with `diff` to analyze what you've done and to help debug any issues.\n"
    "7. **Consult Web:** Use `WebFetch` if you need external information.\n"
    "8. **Finish:** Once the task is complete, call `EndTask` with a `reason` of 'success' and `info` summarizing your work. If you cannot complete the task, call `EndTask` with 'failure' and explain why.\n\n"
    "**Tool Guidelines:**\n"
    "- `Think`: Use this to externalize your thought process. It helps you structure your plan and analyze information before taking action.\n"
    "- `UserInput`: Use this to ask for feedback, clarification, or the next feature to implement. Essential for interactive development.\n"
    "- `WriteFile`: Creates a new file or completely overwrites an existing one. Use with caution.\n"
    "- `Edit`: Performs a targeted search-and-replace. This is safer for small changes.\n"
    "- `Git`: Use this to manage version control. `add` and `commit` changes frequently. Use `diff` to review your work and `log` to see history.\n"
    "- `McpListTools` & `McpRunTool`: Use these to interact with external systems via the Model Context Protocol (MCP). First, use `McpListTools` with a configured server name to see what tools it offers. Then, use `McpRunTool` to execute a specific tool with the required parameters.\n"
    "- `Shell`: Executes shell commands. Powerful but dangerous. Use it only when necessary.\n"
    "- `MakeTodoList`, `CheckTodoList`, `MarkTodoItemComplete`: Use these tools to create and manage todo lists for complex tasks. Break down large tasks into smaller items and track progress.\n"
    "- `EndTask`: You MUST call this tool to signal you have finished your task. Provide a clear reason ('success' or 'failure') and a summary of your work in the 'info' field.\n\n"
    "Always use relative paths. Be methodical. Think step by step."
)

ASK_SYSTEM_PROMPT = (
    "You are an AI assistant designed to answer questions and provide information. You are in 'ask' mode.\n"
    "Your goal is to be a helpful and knowledgeable resource. You can read files and browse the web to find answers.\n\n"
    "**Workflow:**\n"
    "1. **Check Memories:** Consult your PERMANENT MEMORIES first to see if the answer is already there.\n"
    "2. **Understand the Question:** Analyze the user's query to fully grasp what they are asking.\n"
    "3. **Gather Information:** If memories are insufficient, use `ReadFile` to examine relevant files and `WebFetch` to get external information if necessary.\n"
    "4. **Synthesize and Answer:** Combine the information you've gathered to provide a comprehensive and clear answer. Use `Think` to structure your thoughts.\n\n"
    "**Tool Guidelines:**\n"
    "- You do **not** have access to tools that modify files (`WriteFile`, `Edit`) or execute shell commands (`Shell`).\n"
    "- `McpListTools` & `McpRunTool`: Use these to interact with external systems via the Model Context Protocol (MCP) to gather information.\n"
    "- Focus on providing information and answering questions."
)

ARCHITECT_SYSTEM_PROMPT = (
    "You are a principal AI software architect. You are in 'architect' mode.\n"
    "Your purpose is to create high-level plans and designs for software projects. Do NOT write implementation code.\n\n"
    "**Workflow:**\n"
    "1. **Check Memories:** Consult your PERMANENT MEMORIES first for existing architectural context.\n"
    "2. **Gather Context:** If memories are insufficient, use `ReadFolder` and `ReadFile` to understand the existing project structure and code.\n"
    "3. **Deconstruct the Request:** Use the `Think` tool to break down the user's request into architectural components and requirements.\n"
    "4. **Design the Plan:** Formulate a detailed, step-by-step plan. Describe new files to be created, changes to existing files, and the overall structure. Do not write the code for these changes.\n"
    "5. **Seek Clarification:** Use `UserInput` if the requirements are ambiguous or to get feedback on your proposed plan.\n"
    "6. **Finish:** When your plan is complete, call the `EndTask` tool. Set `reason` to 'success' and put your complete, final plan into the `info` parameter.\n\n"
    "**Tool Guidelines:**\n"
    "- Your primary tool is `Think` to outline your architectural plan.\n"
    "- You can read files, but you cannot write or edit them. You cannot use `Shell`.\n"
    "- `McpListTools` & `McpRunTool`: Use these to inspect external systems or data sources that may influence the architectural plan.\n"
    "- `EndTask`: You MUST use this tool to submit your final plan. Put the complete plan in the 'info' parameter.\n"
    "- Your final output should be a plan, not executable code."
)

AGENT_MAKER_SYSTEM_PROMPT = (
    "You are a master AI agent that can create and delegate tasks to other specialized AI agents.\n"
    "Your purpose is to break down complex requests into sub-tasks that can be handled by other agents.\n\n"
    "**Workflow:**\n"
    "1. **Deconstruct Request:** Use `Think` to analyze the user's request and break it down into a sequence of tasks.\n"
    "2. **Delegate:** For each task, use the `make_subagent` tool. Assign the correct mode ('code' or 'architect') and provide a clear, specific prompt for the sub-agent.\n"
    "3. **Synthesize:** Combine the results from the sub-agents to fulfill the original request.\n"
    "4. **Consult:** Use `ReadFolder`, `ReadFile`, and `WebFetch` to gather any information needed to create effective prompts for your sub-agents.\n\n"
    "**Tool Guidelines:**\n"
    "- `make_subagent`: Your primary tool for creating other agents. The tool returns a JSON string with 'reason' and 'info' fields. You must check the 'reason' to see if the sub-agent succeeded. If it failed, you may need to debug or create a new sub-agent with a corrected prompt.\n"
    "- `McpListTools` & `McpRunTool`: Use these to interact with external systems to gather information required for creating sub-agent prompts.\n"
    "- `MakeTodoList`, `CheckTodoList`, `MarkTodoItemComplete`: Use these tools to create and manage todo lists for complex multi-step processes. Break down large tasks into smaller items and track progress.\n"
    "- You do not write code or perform edits directly. You delegate these tasks.\n"
    "- Forbidden sub-agent modes: 'ask', 'agent-maker'."
)

MEMORY_SYSTEM_PROMPT = (
    "You are an AI assistant specializing in software analysis. You are in 'memory' mode.\n"
    "Your purpose is to create a comprehensive, structured summary of a codebase and save it to memory for future reference by other agents. You do not answer questions directly; your goal is to create one high-quality, consolidated memory for the project.\n\n"
    "**Mandatory Workflow:**\n"
    "1. **Explore:** Use `ReadFolder` recursively on all directories to map out the entire project structure.\n"
    "2. **Analyze Code:** Use `ReadFile` to read the contents of all relevant source code files. You must understand the full picture before summarizing.\n"
    "3. **Synthesize and Structure the Memory:** Your primary goal is to create a single, well-organized markdown document. Use the `Think` tool to consolidate your findings. The final output MUST contain the following two sections in order:\n"
    "    a. **Architecture Diagram:** A Mermaid diagram (inside a `mermaid` code block) that illustrates the application's architecture, data flow, or component interactions. Choose the most appropriate diagram type (e.g., graph, flowchart).\n"
    "    b. **File Manifest:** A detailed list of all important files. For each file, provide:\n"
    "        - A brief explanation of its purpose.\n"
    "        - A summary of its most important functions/classes, including their purpose and arguments (e.g., `function_name(arg1: type, arg2: type)`).\n"
    "4. **Save:** Once you have composed the complete memory document (diagram + manifest), use the `SaveMemory` tool a single time to save the entire markdown document to the 'project' scope.\n\n"
    "**Tool Guidelines:**\n"
    "- Your final output is a SINGLE call to `SaveMemory` with the complete, structured text.\n"
    "- Use `scope='project'`. Do not use `scope='global'`.\n"
    "- You cannot modify files (`WriteFile`, `Edit`) or execute shell commands (`Shell`).\n"
    "- Do not provide conversational answers. Your entire process should build towards the final `SaveMemory` call."
)

SYSTEM_PROMPTS = {
    "code": CODE_SYSTEM_PROMPT,
    "ask": ASK_SYSTEM_PROMPT,
    "architect": ARCHITECT_SYSTEM_PROMPT,
    "agent-maker": AGENT_MAKER_SYSTEM_PROMPT,
    "memory": MEMORY_SYSTEM_PROMPT,
}

def is_config_valid(cfg):
    """
    Checks if the config is valid.
    It's valid if the 'global' mode is configured, or if at least one
    other mode is configured. A mode is considered configured if it
    has a provider and a model. The API key is optional.
    """
    modes_cfg = cfg.get("modes", {})
    if not modes_cfg:
        return False

    # Helper to check if a single mode's config is valid
    def _is_mode_section_valid(mode_section):
        active_provider = mode_section.get("active_provider")
        if not active_provider:
            return False
        provider_config = mode_section.get("providers", {}).get(active_provider, {})
        # Model is required, but API key is not.
        return "model" in provider_config

    # 1. Check if global config is valid, which makes the whole config usable
    if "global" in modes_cfg and _is_mode_section_valid(modes_cfg["global"]):
        return True

    # 2. If global is not set, check if any other mode is validly configured
    for mode_name, mode_config in modes_cfg.items():
        if mode_name == "global":
            continue
        if _is_mode_section_valid(mode_config):
            return True
            
    return False


def _get_available_tools(agent_mode: str, is_sub_agent: bool, cfg: dict) -> list:
    """Gets the list of available tools metadata based on the agent's mode and config."""
    tools_settings = cfg.get("tools_settings", {})
    enable_user_input = tools_settings.get("enable_user_input", False)
    enable_think = tools_settings.get("enable_think", True)

    disallowed_tools = set()

    if not enable_think:
        disallowed_tools.add("Think")
    if not enable_user_input:
        disallowed_tools.add("UserInput")

    # The 'memory' mode is the only one that should be able to save memories.
    if agent_mode != "memory":
        disallowed_tools.add("SaveMemory")

    if agent_mode in ["ask", "memory", "architect"]:
        disallowed_tools.update({"WriteFile", "Edit", "Shell"})
    elif agent_mode == "agent-maker":
        # Agent-maker can only read, think, make sub-agents, and manage todo lists.
        disallowed_tools.update({"WriteFile", "Edit", "Shell", "UserInput"})

    # Sub-agents have additional restrictions
    if is_sub_agent:
        disallowed_tools.update({"UserInput", "MakeSubagent"})
    else: # Non-sub-agents cannot end the task
        disallowed_tools.add("EndTask")
    
    # All modes except agent-maker cannot create sub-agents.
    if agent_mode != "agent-maker":
        disallowed_tools.add("MakeSubagent")

    return [
        t for t in tools.TOOLS_METADATA if t["function"]["name"] not in disallowed_tools
    ]

def _get_model_info_for_mode(cfg: dict, agent_mode: str) -> tuple:
    """Gets model info (name, capabilities) for a given agent mode."""
    modes = cfg.get("modes", {})
    global_config = modes.get("global", {})
    mode_config = modes.get(agent_mode, {})
    active_provider = mode_config.get("active_provider") or global_config.get("active_provider")
    
    if not active_provider:
        return None, None

    global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
    mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
    final_provider_config = {**global_provider_settings, **mode_provider_settings}
    
    model_name = final_provider_config.get("model")
    if not model_name:
        return None, None

    provider_for_litellm = "openai" if active_provider == "hackclub_ai" else active_provider
    model_str = f"{provider_for_litellm}/{model_name}"
    
    all_models_info = config._get_provider_models()
    lookup_provider = "openai" if active_provider == "hackclub_ai" else active_provider
    model_capabilities = all_models_info.get(lookup_provider, {}).get(model_name, {})
    
    return model_str, model_capabilities

def _clear_llm_project_memories():
    """Removes memories generated by the LLM from the project memory file."""
    try:
        project_name = os.path.basename(os.getcwd())
        project_memory_file = config.DATA_DIR / f"{project_name}.md"
        
        if not project_memory_file.is_file():
            return

        content = project_memory_file.read_text(encoding="utf-8")
        
        # Memories are separated by ---. Split and filter.
        memories = content.split("\n\n---\n\n")
        
        user_memories = [
            mem for mem in memories if mem.strip() and "<!-- USER Generated Memory -->" in mem
        ]

        new_content = "\n\n---\n\n".join(user_memories)
        project_memory_file.write_text(new_content, encoding="utf-8")
    except Exception as e:
        console.print(f"[bold red]Error clearing LLM memories:[/] {e}")


def _run_memory_update(cfg, lock, in_background):
    """Clears LLM-generated project memories and runs a sub-agent to regenerate them."""
    if not lock.acquire(blocking=False):
        if in_background:
            # Show temporary message in hotbar for 3 seconds
            print("\033[s\033[2K\033[1;33mBackground memory update skipped (already in progress)\033[0m\033[u", end="", flush=True)
            import time
            time.sleep(3)
            print("\033[s\033[2K\033[u", end="", flush=True)
        else:
            console.print("[bold yellow]Skipping memory update as another is already in progress.[/bold yellow]")
        return

    def update_task():
        try:
            _clear_llm_project_memories()
            
            run_sub_agent(
                mode="memory",
                prompt="Make memories for this project",
                cfg=cfg
            )
        finally:
            lock.release()
            if in_background:
                # Show temporary message in hotbar for 3 seconds
                print("\033[s\033[2K\033[1;32mBackground memory update finished\033[0m\033[u", end="", flush=True)
                import time
                time.sleep(3)
                print("\033[s\033[2K\033[u", end="", flush=True)

    if in_background:
        thread = threading.Thread(target=update_task)
        thread.start()
    else:
        with console.status("[bold yellow]Updating memories...[/]"):
            update_task()

def _run_rag_update(rag_retriever, rag_settings, lock, in_background):
    if not lock.acquire(blocking=False):
        if in_background:
            # Show temporary message in hotbar for 3 seconds
            print("\033[s\033[2K\033[1;33mBackground RAG update skipped (already in progress)\033[0m\033[u", end="", flush=True)
            import time
            time.sleep(3)
            print("\033[s\033[2K\033[u", end="", flush=True)
        else:
            console.print("[bold yellow]Skipping RAG update as another is already in progress.[/bold yellow]")
        return

    def update_task():
        try:
            rag_retriever.index_project(
                batch_size=rag_settings.get("rag_batch_size", 100), 
                force_reindex=True, 
                quiet=in_background
            )
        finally:
            lock.release()
            if in_background:
                # Show temporary message in hotbar for 3 seconds
                print("\033[s\033[2K\033[1;32mBackground RAG update finished\033[0m\033[u", end="", flush=True)
                import time
                time.sleep(3)
                print("\033[s\033[2K\033[u", end="", flush=True)

    if in_background:
        thread = threading.Thread(target=update_task)
        thread.start()
    else:
        with console.status("[bold yellow]Updating RAG index...[/]"):
            update_task()

def _get_sub_agent_system_prompt(mode: str, cfg: dict) -> str:
    """Builds the system prompt for a sub-agent."""
    modes_cfg = cfg.get("modes", {})
    global_cfg = modes_cfg.get("global", {})
    mode_cfg = modes_cfg.get(mode, {})
    # Get tool strategy from mode > global > default
    tool_strategy = mode_cfg.get("tool_strategy") or global_cfg.get("tool_strategy") or "tool_calls"

    memories = load_memories()
    system_prompt_template = SYSTEM_PROMPTS.get(mode, CODE_SYSTEM_PROMPT)
    custom_instructions = cfg.get("custom_instructions", "")
    
    system_prompt_parts = []
    if tool_strategy == "xml":
        available_tools = _get_available_tools(mode, is_sub_agent=True, cfg=cfg)
        system_prompt_parts.append(generate_xml_tool_prompt(available_tools))

    if custom_instructions:
        system_prompt_parts.append(f"### CUSTOM INSTRUCTIONS ###\n{custom_instructions}")

    if memories:
        system_prompt_parts.append(f"### PERMANENT MEMORIES ###\n{memories}")
    
    system_prompt_parts.append(f"### TASK ###\n{system_prompt_template}")
    return "\n\n".join(filter(None, system_prompt_parts))


def run_sub_agent(mode: str, prompt: str, cfg: dict) -> str:
    """
    Runs a non-interactive sub-agent for a specific task.
    Returns a JSON string with the result from the sub-agent.
    """
    console.print(Panel(f"Starting sub-agent in '{mode}' mode...\nPrompt: {prompt}", title="[bold blue]Sub-agent Invoked[/]", border_style="blue"))

    system_prompt = _get_sub_agent_system_prompt(mode, cfg)
    sub_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    sub_read_files = set()

    try:
        # Run the agent loop. It will end by either returning a message (error)
        # or raising SubAgentEndTask (success/failure).
        sub_agent_session_stats = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "last_prompt_tokens": 0, "edit_count": 0}
        final_message = process_llm_turn(sub_messages, sub_read_files, cfg, mode, session_stats=sub_agent_session_stats, yolo_mode=False, is_sub_agent=True)
        
        if final_message:
            output_content = final_message.get("content", "Sub-agent did not return any output.")
            info = f"Last Message of Subagent: {output_content}\nA sub-agent must end its task by calling the 'EndTask' tool. It should not return a final text response. You should make a new agent to try again."
            output = json.dumps({"reason": "Model Error", "info": info})
        else:
            # This happens if mode is not configured, process_llm_turn returns None
            info = f"Sub-agent failed to start. The '{mode}' mode might not be configured correctly."
            output = json.dumps({"reason": "Configuration Error", "info": info})

    except SubAgentEndTask as e:
        output = json.dumps({"reason": e.reason, "info": e.info})
    
    console.print(Panel(json.dumps(json.loads(output), indent=2), title="[bold blue]Sub-agent Finished[/]", border_style="blue"))
    return output


def _should_add_to_history(text: str):
    """Return True if the given input text should be added to history."""
    text = text.strip()
    # Don't save empty lines or commands to history
    if not text or text.startswith(('/', '!')) or text.lower() == "exit":
        return False
    return True


def process_llm_turn(messages, read_files_in_session, cfg, agent_mode: str, session_stats: dict, yolo_mode: bool = False, is_sub_agent: bool = False):
    """Handles a single turn of the LLM, including tool calls and user confirmation."""
    DANGEROUS_TOOLS = {"WriteFile", "Edit", "Shell", "Git"}
    available_tools_metadata = _get_available_tools(agent_mode, is_sub_agent, cfg)
    show_reasoning = cfg.get("misc_settings", {}).get("show_reasoning", False)

    # Get model and API key, falling back to global settings
    modes = cfg.get("modes", {})
    global_config = modes.get("global", {})
    mode_config = modes.get(agent_mode, {})

    # Check for temporary model override
    temp_model = cfg.get("temp_model")
    if temp_model:
        active_provider = temp_model["provider"]
        model_name = temp_model["model"]
        api_key = _find_api_key_for_provider(cfg, active_provider)
        # For temporary models, use defaults
        tool_strategy = "tool_calls"
        enable_web_search = False
        api_base = None
        if model_name:
            provider_for_litellm = "openai" if active_provider == "hackclub_ai" else active_provider
            model = f"{provider_for_litellm}/{model_name}"
    else:
        active_provider = mode_config.get("active_provider") or global_config.get("active_provider")
        model, api_key, api_base = None, None, None
        
        # Get tool strategy from mode > global > default
        tool_strategy = mode_config.get("tool_strategy") or global_config.get("tool_strategy") or "tool_calls"
        enable_web_search = False

        if active_provider:
            # Mode-specific provider settings override global ones
            global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
            mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
            
            # Merge settings, with mode-specific taking precedence
            final_provider_config = {**global_provider_settings, **mode_provider_settings}
            
            model_name = final_provider_config.get("model")
            api_key = final_provider_config.get("api_key") # Can be None
            
            # Special handling for hackclub_ai
            provider_for_litellm = "openai" if active_provider == "hackclub_ai" else active_provider
            api_base = None if active_provider == "hackclub_ai" else final_provider_config.get("api_base")

            enable_web_search = final_provider_config.get("enable_web_search", False)
            
            if model_name:
                model = f"{provider_for_litellm}/{model_name}"

    if not model: # Model is required, API key is not
        console.print(f"[bold red]Error:[/] Agent mode '{agent_mode}' is not configured (or is missing a model).")
        console.print("Please use `/config` to set the provider and model for this mode or for the 'global' settings.")
        return # Stop processing and return to the user prompt

    all_models_info = config._get_provider_models()
    # For hackclub_ai, we look up capabilities using 'openai' as the key
    lookup_provider = "openai" if active_provider == "hackclub_ai" else active_provider
    model_capabilities = all_models_info.get(lookup_provider, {}).get(model_name, {})
    supports_system_message = model_capabilities.get("supports_system_message", True)

    # If model doesn't support system messages, move content to the first user message.
    if not supports_system_message and messages and messages[0]["role"] == "system":
        system_message = messages.pop(0)
        # Find the first user message to prepend to
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = f"SYSTEM INSTRUCTIONS:\n{system_message['content']}\n\n---\n\n{msg['content']}"
                break

    while True:
        if tool_strategy == 'tool_calls':
            response = litellm.completion(
                model=model,
                api_key=api_key,
                api_base=api_base,
                messages=messages,
                tools=available_tools_metadata,
                tool_choice="auto",
                search=enable_web_search,
            )
            _update_session_stats(response, session_stats, model_capabilities)
            choice = response.choices[0]
            if show_reasoning and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                console.print(
                    Panel(
                        Markdown(choice.message.reasoning_content, style="default", code_theme="monokai"),
                        title="[bold blue]Reasoning[/]",
                        border_style="blue",
                    )
                )
            if choice.finish_reason != "tool_calls":
                break # Go to final streaming response
            
            messages.append(choice.message)
            tool_calls = choice.message.tool_calls
        else: # xml strategy
            response = litellm.completion(model=model, api_key=api_key, api_base=api_base, messages=messages, search=enable_web_search)
            _update_session_stats(response, session_stats, model_capabilities)
            choice = response.choices[0]
            if show_reasoning and hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                console.print(
                    Panel(
                        Markdown(choice.message.reasoning_content, style="default", code_theme="monokai"),
                        title="[bold blue]Reasoning[/]",
                        border_style="blue",
                    )
                )
            response_content = choice.message.content or ""
            
            # Some models (like qwen) add thinking tags. Strip them for cleaner output.
            response_content = re.sub(r'<thinking-content-.*?>', '', response_content, flags=re.DOTALL)

            # Separate text from tool calls
            tool_xml_blocks = re.findall(r'<tool_code>(.*?)</tool_code>', response_content, re.DOTALL)
            text_content = re.sub(r'<tool_code>.*?</tool_code>', '', response_content, flags=re.DOTALL).strip()
            
            # If there's conversational text, display it immediately so the user sees the agent's reasoning.
            if text_content:
                panel = Panel(
                    Markdown(text_content, style="default", code_theme="monokai"),
                    title="[bold green]Assistant[/]",
                    border_style="green",
                )
                console.print(panel)

            # Add the full, unmodified response to history for the model's context.
            messages.append(choice.message)

            if not tool_xml_blocks:
                # The response was just text, which we've displayed. We're done for this turn.
                return messages[-1]
            
            # Convert XML blocks to a format that resembles native tool_calls
            tool_calls = []
            for tool_xml in tool_xml_blocks:
                try:
                    root = ET.fromstring(f"<root_tool>{tool_xml.strip()}</root_tool>")
                    tool_call_element = root[0]
                    tool_name = tool_call_element.tag
                    tool_args = {child.tag: child.text for child in tool_call_element}
                    # Create a mock tool_call object to unify processing
                    tool_calls.append({
                        "id": f"xml_call_{os.urandom(8).hex()}",
                        "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        "type": "function",
                    })
                except ET.ParseError as e:
                    console.print(f"[bold red]XML Parse Error for tool call:[/]\n{tool_xml}\nError: {e}")
                    continue
        
        # --- Cost Confirmation Logic ---
        safety_settings = cfg.get("safety_settings", {})
        cost_threshold = safety_settings.get("cost_threshold", 0.0)
        
        if cost_threshold > 0.0:
            estimated_cost = 0.0
            sub_agent_calls_to_confirm = []
            
            for tool_call in tool_calls:
                is_native_call = hasattr(tool_call, 'function')
                tool_name = tool_call.function.name if is_native_call else tool_call["function"]["name"]
                if tool_name == "MakeSubagent":
                    try:
                        arguments_str = tool_call.function.arguments if is_native_call else tool_call["function"]["arguments"]
                        sub_agent_calls_to_confirm.append(json.loads(arguments_str))
                    except json.JSONDecodeError:
                        continue # Ignore malformed args for cost check
            
            if sub_agent_calls_to_confirm:
                for call_args in sub_agent_calls_to_confirm:
                    sub_agent_mode = call_args.get("mode")
                    sub_agent_prompt = call_args.get("prompt")
                    if not sub_agent_mode or not sub_agent_prompt: continue

                    model_str, model_caps = _get_model_info_for_mode(cfg, sub_agent_mode)
                    if not model_str or not model_caps: continue
                    
                    system_prompt = _get_sub_agent_system_prompt(sub_agent_mode, cfg)
                    sub_agent_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sub_agent_prompt}
                    ]
                    
                    in_cost = model_caps.get("input_cost_per_token", 0) or 0
                    try:
                        prompt_tokens = litellm.token_counter(model=model_str, messages=sub_agent_messages)
                        estimated_cost += prompt_tokens * in_cost
                    except Exception:
                        pass # Cannot estimate cost for this call

            if estimated_cost > cost_threshold:
                if not Confirm.ask(f"[bold yellow]Executing tool(s) has an estimated minimum cost of ${estimated_cost:.4f}. Proceed?[/]"):
                    console.print("[bold red]Tool execution cancelled by user due to cost.[/bold red]")
                    # Remove the assistant's message that contained the expensive tool call
                    if messages[-1]["role"] == "assistant":
                        messages.pop()
                    
                    # Add user feedback message and re-prompt the LLM
                    user_feedback = "User cancelled tool execution due to high estimated cost. Please reconsider your plan, find a cheaper alternative, or ask the user to proceed."
                    messages.append({"role": "user", "content": user_feedback})
                    continue # Restart the while-true loop in process_llm_turn

        # --- Common Tool Execution Logic ---
        has_executed_tool = False
        xml_tool_outputs = []

        for tool_call in tool_calls:
            # For XML, tool_call is a dict; for native, it's an object. Access attrs consistently.
            is_native_call = hasattr(tool_call, 'function')
            tool_name = tool_call.function.name if is_native_call else tool_call["function"]["name"]
            
            try:
                arguments_str = tool_call.function.arguments if is_native_call else tool_call["function"]["arguments"]
                tool_args = json.loads(arguments_str)
            except json.JSONDecodeError:
                console.print(f"[bold red]Error:[/] Could not decode arguments for tool '{tool_name}': {arguments_str}")
                continue

            # SPECIAL HANDLING for EndTask - this will terminate the sub-agent
            if tool_name == "EndTask":
                raise SubAgentEndTask(
                    reason=tool_args.get("reason"),
                    info=tool_args.get("info", "")
                )

            tool_panel_content = f"[cyan]{tool_name}[/][default]({json.dumps(tool_args, indent=2)})[/]"
            console.print(
                Panel(
                    tool_panel_content,
                    title="[bold yellow]Tool Call[/]",
                    border_style="yellow",
                    expand=False,
                )
            )
            
            tool_output = "" # Initialize tool_output

            # SPECIAL HANDLING FOR MakeSubagent
            if tool_name == "MakeSubagent":
                tool_output = run_sub_agent(
                    mode=tool_args["mode"],
                    prompt=tool_args["prompt"],
                    cfg=cfg
                )
            # SPECIAL HANDLING for interactive tools that shouldn't be in a spinner
            elif tool_name == "UserInput":
                if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                    tool_output = tool_func(**tool_args)
                else:
                    tool_output = f"Unknown tool '{tool_name}'"
            # REGULAR TOOL EXECUTION
            elif tool_name in DANGEROUS_TOOLS and not yolo_mode:
                if not Confirm.ask(
                    f"[bold yellow]Execute the [cyan]{tool_name}[/cyan] tool with the arguments above?[/]",
                    default=True
                ):
                    console.print("[bold red]Skipping tool call.[/]")
                    tool_output = "User denied execution of this tool call."
                else:
                    if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                        if tool_name in ["WriteFile", "Edit"]:
                            session_stats["edit_count"] = session_stats.get("edit_count", 0) + 1
                        # Inject session-specific state if needed by the tool
                        if tool_name in ["ReadFile", "ReadManyFiles"]:
                            tool_args["read_files_in_session"] = read_files_in_session
                            tool_args["messages"] = messages
                        elif tool_name == "ReadSymbol":
                            # Initialize symbol tracking if not already present
                            if not hasattr(read_files_in_session, 'symbols'):
                                read_files_in_session.symbols = {}
                            tool_args["read_symbols_in_session"] = read_files_in_session.symbols
                        with console.status("[bold yellow]Executing tool..."):
                            tool_output = tool_func(**tool_args)
                    else:
                        tool_output = f"Unknown tool '{tool_name}'"
            else: # Not dangerous or YOLO mode is on
                if tool_name in DANGEROUS_TOOLS and not yolo_mode:
                    # This case handles when yolo_mode is False but we still need confirmation
                    if not Confirm.ask(
                        f"[bold yellow]Execute the [cyan]{tool_name}[/cyan] tool with the arguments above?[/]",
                        default=True
                    ):
                        console.print("[bold red]Skipping tool call.[/]")
                        tool_output = "User denied execution of this tool call."
                    else:
                        if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                            if tool_name in ["WriteFile", "Edit"]:
                                session_stats["edit_count"] = session_stats.get("edit_count", 0) + 1
                            if tool_name in ["ReadFile", "ReadManyFiles"]:
                                tool_args["read_files_in_session"] = read_files_in_session
                                tool_args["messages"] = messages
                            elif tool_name == "ReadSymbol":
                                # Initialize symbol tracking if not already present
                                if not hasattr(read_files_in_session, 'symbols'):
                                    read_files_in_session.symbols = {}
                                tool_args["read_symbols_in_session"] = read_files_in_session.symbols
                            with console.status("[bold yellow]Executing tool..."):
                                tool_output = tool_func(**tool_args)
                        else:
                            tool_output = f"Unknown tool '{tool_name}'"
                else:
                    # YOLO mode is on or tool is not dangerous - execute without confirmation
                    if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                        if tool_name in ["WriteFile", "Edit"]:
                            session_stats["edit_count"] = session_stats.get("edit_count", 0) + 1
                        if tool_name in ["ReadFile", "ReadManyFiles"]:
                            tool_args["read_files_in_session"] = read_files_in_session
                            tool_args["messages"] = messages
                        elif tool_name == "ReadSymbol":
                            # Initialize symbol tracking if not already present
                            if not hasattr(read_files_in_session, 'symbols'):
                                read_files_in_session.symbols = {}
                            tool_args["read_symbols_in_session"] = read_files_in_session.symbols
                        with console.status("[bold yellow]Executing tool..."):
                            tool_output = tool_func(**tool_args)
                    else:
                        tool_output = f"Unknown tool '{tool_name}'"
            
            # --- Append tool output based on strategy ---
            if tool_strategy == "xml":
                xml_tool_outputs.append(f"<tool_output tool_name='{tool_name}'>\n{str(tool_output)}\n</tool_output>")
            else: # Native tool_calls strategy
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id if is_native_call else tool_call["id"],
                    "name": tool_name,
                    "content": str(tool_output),
                })
            
            has_executed_tool = True

        if has_executed_tool:
            if tool_strategy == "xml" and xml_tool_outputs:
                combined_output = "\n\n".join(xml_tool_outputs)
                user_prompt = f"The tool calls produced the following output:\n\n{combined_output}\n\nBased on this, please continue with your plan."
                messages.append({"role": "user", "content": user_prompt})
            continue
        else: # No tools were run, break to final response
            break

    # This part handles the final text response from the assistant.
    # For the XML strategy, final responses are handled inside the loop.
    # This block is for the `tool_calls` strategy's final streaming response.
    if tool_strategy == 'tool_calls':
        full_response = ""
        panel = Panel(
            "",
            title="[bold green]Assistant[/]",
            border_style="green",
        )
        with Live(panel, refresh_per_second=10, console=console) as live:
            # For streaming, we manually count tokens and update stats
            in_cost = model_capabilities.get("input_cost_per_token", 0) or 0
            out_cost = model_capabilities.get("output_cost_per_token", 0) or 0
            
            try:
                current_prompt_tokens = litellm.token_counter(model=model, messages=messages)
                new_prompt_tokens = current_prompt_tokens - session_stats.get("last_prompt_tokens", 0)
                session_stats["last_prompt_tokens"] = current_prompt_tokens
                
                session_stats['prompt_tokens'] += new_prompt_tokens
                session_stats['cost'] += new_prompt_tokens * in_cost
            except Exception:
                pass # litellm might not know the model, ignore for now

            # For tool_calls, we need a new completion call to get the final answer.
            stream_response = litellm.completion(
                model=model, api_key=api_key, api_base=api_base, messages=messages, stream=True, search=enable_web_search
            )
            for chunk in stream_response:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    live.update(
                        Panel(
                            Markdown(full_response, style="default", code_theme="monokai"),
                            title="[bold green]Assistant[/]",
                            border_style="green",
                        )
                    )
        
        try:
            completion_tokens = litellm.token_counter(model=model, text=full_response)
            session_stats['completion_tokens'] += completion_tokens
            session_stats['cost'] += completion_tokens * out_cost
        except Exception:
            pass
        messages.append({"role": "assistant", "content": full_response})

    # For interactive mode (not sub-agent), return the message directly
    if is_sub_agent:
        return messages[-1]
    else:
        # Just return None for interactive mode to indicate normal completion
        return None

def display_help():
    """Displays the help menu for interactive commands."""
    help_text = """
| Command                  | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `/help`                  | Show this help message.                                          |
| `/config`                | Open the configuration menu.                                     |
| `/scaffold <prompt>`     | Creates a new project structure based on a description.          |
| `/load_preset <name>`    | Load a saved configuration preset.                               |
| `/mode <name>`           | Switch agent mode (code, ask, architect).                        |
| `/clear`                 | Clears the current conversation context.                         |
| `/compress`              | Summarizes the conversation to reduce context size.              |
| `/yolo`                  | Toggle YOLO mode (disables safety confirmations).                |
| `/rag init`              | Initialize RAG for the project (if not already indexed).         |
| `/rag update`            | Force re-indexing of the project files for RAG.                  |
| `/rag deinit`            | Deactivate RAG for the current session.                          |
| `/memory init`           | Loads memories into the current session's context.               |
| `/memory deinit`         | Unloads memories from the current session's context.             |
| `/memory save <text>`    | Saves user-generated text to the project's memory file.          |
| `/memory update`         | Force re-generation of project memories (LLM-generated only).    |
| `/memory delete`         | Deletes the current project's memory file.                       |
| `/model [provider/model]`| Temporarily switch to a different model. Use `/model default` to reset. |
| `/model`                 | Show the current model configuration.                            |
| `/exit` or `exit`        | Exit the interactive session.                                    |
| `! <command>`            | Execute a shell command directly from your terminal.             |

**Hotkeys:**
- **Enter**: Send your message to the agent.
- **Alt+Enter** or **Ctrl+Enter**: Add a new line to your message.
"""
    console.print(
        Panel(
            Markdown(help_text),
            title="[bold cyan]Help Menu[/]",
            border_style="cyan",
            expand=False,
        )
    )


def _find_api_key_for_provider(cfg: dict, provider: str) -> str | None:
    """Searches the config for an API key for the given provider."""
    modes_cfg = cfg.get("modes", {})
    # Prioritize global config
    if "global" in modes_cfg:
        key = modes_cfg["global"].get("providers", {}).get(provider, {}).get("api_key")
        if key:
            return key
    # Then check other modes
    for mode_config in modes_cfg.values():
        key = mode_config.get("providers", {}).get(provider, {}).get("api_key")
        if key:
            return key
    return None

def load_memories() -> str:
    """Loads global and project-specific memories from the data directory."""
    memory_parts = []

    # 1. Load global memory
    global_memory_file = config.DATA_DIR / "memorys.global.md"
    if global_memory_file.is_file():
        try:
            memory_parts.append(f"### Global Memory ###\n{global_memory_file.read_text()}")
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Could not load global memory file: {e}")

    # 2. Load project-specific memory
    try:
        project_name = os.path.basename(os.getcwd())
        project_memory_file = config.DATA_DIR / f"{project_name}.md"
        if project_memory_file.is_file():
            memory_parts.append(f"### Project Memory ({project_name}) ###\n{project_memory_file.read_text()}")
    except Exception as e:
        # This could happen if getcwd fails, though unlikely.
        console.print(f"[bold yellow]Warning:[/] Could not determine or load project memory: {e}")

    if not memory_parts:
        return ""

    return "\n\n".join(memory_parts)

MODES = ["code", "ask", "architect", "agent-maker", "memory"]


def start_interactive_session(initial_prompt, cfg):
    """Runs the agent in interactive mode."""
    agent_mode = "code"
    memories_active = False # Default to OFF
    last_rag_context = ""

    def get_system_prompt(mode, current_cfg):
        # Determine tool strategy from config
        modes_cfg = current_cfg.get("modes", {})
        global_cfg = modes_cfg.get("global", {})
        mode_cfg = modes_cfg.get(mode, {})
        
        # Check for temporary model override
        temp_model = current_cfg.get("temp_model")
        if temp_model:
            # Use temporary model settings
            active_provider = temp_model["provider"]
            model_name = temp_model["model"]
            # For temporary models, default to tool_calls strategy
            tool_strategy = "tool_calls"
        else:
            # Get tool strategy from mode > global > default
            tool_strategy = mode_cfg.get("tool_strategy") or global_cfg.get("tool_strategy") or "tool_calls"
        
        memories = ""
        if memories_active:
            memories = load_memories()

        system_prompt_template = SYSTEM_PROMPTS[mode]
        custom_instructions = current_cfg.get("custom_instructions", "")
        
        system_prompt_parts = []
        if tool_strategy == "xml":
            available_tools = _get_available_tools(mode, is_sub_agent=False, cfg=current_cfg)
            system_prompt_parts.append(generate_xml_tool_prompt(available_tools))

        if custom_instructions:
            system_prompt_parts.append(f"### CUSTOM INSTRUCTIONS ###\n{custom_instructions}")

        if memories:
            system_prompt_parts.append(f"### PERMANENT MEMORIES ###\n{memories}")
        
        if last_rag_context:
            rag_prompt_addition = (
                "### RAG Context From Last Query\n\n"
                "Use the following code context to answer the user's last question.\n\n"
                f"{last_rag_context}"
            )
            system_prompt_parts.append(rag_prompt_addition)

        system_prompt_parts.append(f"### TASK ###\n{system_prompt_template}")
        return "\n\n".join(filter(None, system_prompt_parts))

    messages = [{"role": "system", "content": get_system_prompt(agent_mode, cfg)}]
    _update_environment_for_mode(agent_mode, cfg)
    read_files_in_session = set()
    history = InMemoryHistory()
    yolo_mode = False
    rag_retriever = None
    prompt_count_since_rag_update = 0
    prompt_count_since_memory_update = 0
    rag_update_in_progress = threading.Lock()
    memory_update_in_progress = threading.Lock()
    session_stats = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "last_prompt_tokens": 0, "edit_count": 0}

    # --- Pre-calculate system prompt cost ---
    modes = cfg.get("modes", {})
    global_config = modes.get("global", {})
    mode_config = modes.get(agent_mode, {})
    active_provider = mode_config.get("active_provider") or global_config.get("active_provider")

    if active_provider:
        # This logic is duplicated from the toolbar, but necessary to get model info before the loop
        global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
        mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
        final_provider_config = {**global_provider_settings, **mode_provider_settings}
        model_name = final_provider_config.get("model")

        if model_name:
            all_models_info = config._get_provider_models()
            lookup_provider = "openai" if active_provider == "hackclub_ai" else active_provider
            model_capabilities = all_models_info.get(lookup_provider, {}).get(model_name, {})
            in_cost = model_capabilities.get("input_cost_per_token", 0) or 0
            model_str = f"openai/{model_name}" if active_provider == "hackclub_ai" else f"{active_provider}/{model_name}"
            
            try:
                # Count system prompt tokens and add initial cost
                prompt_tokens = litellm.token_counter(model=model_str, messages=messages)
                session_stats['cost'] += prompt_tokens * in_cost
                session_stats['prompt_tokens'] += prompt_tokens
                session_stats['last_prompt_tokens'] = prompt_tokens
            except Exception:
                pass # Ignore if model is not known to litellm

    # --- Auto-init logic ---
    if cfg.get("memory_settings", {}).get("auto_init_memories", False):
        if load_memories():
            memories_active = True
            messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
            console.print("[bold green]Memories are active (found existing memory files).[/bold green]")
        else:
            console.print("[bold yellow]Memory auto-init is on, but no memory files were found.[/bold yellow]")
            console.print("Use the `memory` mode or `/memory save` to create them.")

    if cfg.get("rag_settings", {}).get("auto_init_rag", False):
        embedding_cfg = cfg.get("embedding")
        if not embedding_cfg or not embedding_cfg.get("provider") or not embedding_cfg.get("model"):
            console.print("[bold yellow]Warning:[/] RAG auto-init is on, but no embedding model is configured. Use `/config` to set it.")
        else:
            api_key = _find_api_key_for_provider(cfg, embedding_cfg["provider"])
            embedding_cfg["api_key"] = api_key
            try:
                rag_instance = CodeRAG(
                    project_path=os.getcwd(),
                    config_dir=config.CONFIG_DIR,
                    embedding_config=embedding_cfg,
                    original_openai_api_base=original_openai_api_base
                )
                if rag_instance.has_index():
                    rag_retriever = rag_instance
                    console.print("[bold green]RAG is active (found existing index).[/bold green]")
                else:
                    console.print("[bold yellow]RAG auto-init is on, but no index found for this project.[/bold yellow]")
                    console.print("Run `/rag init` to build the index.")
            except Exception as e:
                console.print(f"[bold red]Error checking for RAG index:[/] {e}")
                rag_retriever = None
    # --- End auto-init logic ---

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(Panel(initial_prompt, title="[bold blue]User[/]", border_style="blue"))
        messages.append({"role": "user", "content": initial_prompt})
        try:
            process_llm_turn(messages, read_files_in_session, cfg, agent_mode, session_stats, yolo_mode=yolo_mode)
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/] {e}")
            messages.pop()

    while True:
        try:
            # --- New prompt with a frame ---
            prompt_buffer = Buffer(
                multiline=True,
                history=history,
            )

            def get_line_prefix(lineno, wrap_count):
                return to_formatted_text(HTML('<b>> </b>'))

            bindings = KeyBindings()
            
            @bindings.add("up", filter=has_history)
            def _(event):
                """Move up in history."""
                event.current_buffer.history_backward()

            @bindings.add("down", filter=has_history)
            def _(event):
                """Move down in history."""
                event.current_buffer.history_forward()

            @bindings.add("c-c")
            def _(event):
                """Handle Ctrl+C as an exit signal."""
                event.app.exit(exception=KeyboardInterrupt)

            @bindings.add("enter")
            def _(event):
                event.app.exit(result=prompt_buffer.text)

            @bindings.add("escape", "enter")
            @bindings.add("c-j")
            def _(event):
                event.current_buffer.insert_text('\n')
            
            # --- Get current mode's settings for toolbar ---
            modes = cfg.get("modes", {})
            global_config = modes.get("global", {})
            mode_config = modes.get(agent_mode, {})
            
            # Check for temporary model override
            temp_model = cfg.get("temp_model")
            if temp_model:
                active_provider = temp_model["provider"]
                model_name = temp_model["model"]
                tool_strategy = "tool_calls"  # Default for temporary models
                enable_web_search = False
            else:
                active_provider = mode_config.get("active_provider") or global_config.get("active_provider")
                model_name, tool_strategy, enable_web_search = None, "tool_calls", False
            
            model_capabilities = {}
            
            if active_provider:
                if temp_model:
                    # For temporary models, we don't have full config, so we use defaults
                    model_name = temp_model["model"]
                    tool_strategy = "tool_calls"
                    enable_web_search = False
                    
                    # Try to get model capabilities from the provider models info
                    all_models_info = config._get_provider_models()
                    lookup_provider = "openai" if active_provider == "hackclub_ai" else active_provider
                    model_capabilities = all_models_info.get(lookup_provider, {}).get(model_name, {})
                else:
                    global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
                    mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
                    final_provider_config = {**global_provider_settings, **mode_provider_settings}
                    
                    model_name = final_provider_config.get("model")
                    tool_strategy = mode_config.get("tool_strategy") or global_config.get("tool_strategy") or "tool_calls"
                    enable_web_search = final_provider_config.get("enable_web_search", False)

                    if model_name:
                        all_models_info = config._get_provider_models()
                        lookup_provider = "openai" if active_provider == "hackclub_ai" else active_provider
                        model_capabilities = all_models_info.get(lookup_provider, {}).get(model_name, {})

            supports_tool_calls = model_capabilities.get("supports_function_calling", False)
            supports_web_search = model_capabilities.get("supports_web_search", False)
            max_input_tokens = model_capabilities.get("max_input_tokens")
            
            tool_status = "✅" if supports_tool_calls and tool_strategy == "tool_calls" else "❌"
            search_status = "✅" if supports_web_search and enable_web_search else "❌"

            # Estimate current context tokens for display
            current_tokens = 0
            if active_provider and model_name:
                try:
                    # Construct model string for litellm
                    model_str = f"openai/{model_name}" if active_provider == "hackclub_ai" else f"{active_provider}/{model_name}"
                    current_tokens = litellm.token_counter(model=model_str, messages=messages)
                except Exception:
                    pass # Ignore if model is not known to litellm

            token_display = ""
            if max_input_tokens:
                token_display = f"Ctx: {current_tokens / 1000:.1f}k/{max_input_tokens / 1000:.0f}k"
            else:
                token_display = f"Ctx: {current_tokens / 1000:.1f}k" if current_tokens > 0 else "Ctx: N/A"

            cost_display = f"Cost: ${session_stats['cost']:.3f}"
            
            # Add model display to toolbar
            model_display = f"{active_provider}/{model_name}" if active_provider and model_name else "No model"
            if temp_model:
                model_display = f"[TEMP] {model_display}"

            input_frame = Frame(
                Window(
                    content=BufferControl(buffer=prompt_buffer),
                    get_line_prefix=get_line_prefix,
                    wrap_lines=True
                ),
                title=to_formatted_text(HTML("<b>Your Turn</b>")),
                style="fg:cyan"
            )

            toolbar_text = f"<b>({agent_mode})</b> {model_display} | {token_display} | {cost_display} | Tools {tool_status} Search {search_status} | <b>[Alt+Enter]</b> new line, <b>/help</b>"
            toolbar = ConditionalContainer(
                Window(
                    content=FormattedTextControl(to_formatted_text(HTML(toolbar_text))),
                    height=1,
                    style="class:bottom-toolbar"
                ),
                filter=~is_done
            )

            layout = Layout(HSplit([input_frame, toolbar]))
            
            app = Application(
                layout=layout,
                key_bindings=bindings,
                mouse_support=True,
                full_screen=False,
            )

            user_input_text = app.run()
            
            if user_input_text is None: # Ctrl+C/D in prompt
                raise EOFError

            if _should_add_to_history(user_input_text):
                history.append_string(user_input_text)

            user_input = user_input_text.strip()
            # --- End new prompt logic ---

            if not user_input:
                continue

            if user_input.lower() == "/help":
                display_help()
                continue
            elif user_input.lower() == "/clear":
                last_rag_context = ""
                messages = [{"role": "system", "content": get_system_prompt(agent_mode, cfg)}]
                read_files_in_session.clear()
                console.print("[bold green]Conversation context has been cleared.[/bold green]")
                continue
            elif user_input.lower() == "/compress":
                weak_model_cfg = cfg.get("weak_model", {})
                provider = weak_model_cfg.get("provider")
                model_name = weak_model_cfg.get("model")

                if not provider or not model_name:
                    console.print("[bold red]Error:[/] Weak model not configured. Use `/config` to set it.")
                    continue

                if len(messages) <= 1:
                    console.print("[yellow]Not enough conversation to compress.[/yellow]")
                    continue

                api_key = _find_api_key_for_provider(cfg, provider)
                model = f"{provider}/{model_name}"
                
                conversation_text_parts = []
                for msg in messages[1:]: # Skip system prompt
                    role = msg.get("role")
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")

                    if role == "user":
                        conversation_text_parts.append(f"User: {str(content)}")
                    elif role == "assistant":
                        text = "Assistant:"
                        if content:
                            text += f"\n{str(content)}"
                        if tool_calls:
                            # Handle both object and dict representations of tool_calls
                            for tc in tool_calls:
                                try:
                                    if isinstance(tc, dict):
                                        func = tc.get("function", {})
                                        name = func.get("name")
                                        args = func.get("arguments")
                                    else: # assume object
                                        name = tc.function.name
                                        args = tc.function.arguments
                                    text += f"\n- Tool Call: {name}({args})"
                                except Exception:
                                    text += "\n- (Could not parse tool call)"
                        conversation_text_parts.append(text)
                    elif role == "tool":
                        tool_name = msg.get("name")
                        tool_content = str(msg.get("content", ""))
                        conversation_text_parts.append(f"Tool ({tool_name}):\n{tool_content}")
                
                conversation_for_summary = "\n\n".join(conversation_text_parts)

                # Get recent git commits to add to the context
                git_log_output = ""
                try:
                    git_log_raw = tools.shell("git log -n 5 --pretty=format:'%h - %s (%cr)'")
                    if "STDOUT" in git_log_raw:
                        log_match = re.search(r'STDOUT:\n(.*?)\nSTDERR:', git_log_raw, re.DOTALL)
                        if log_match and log_match.group(1).strip():
                            git_log_output = log_match.group(1).strip()
                except Exception:
                    pass # Ignore if git log fails
                
                summary_prompt = (
                    "You are a summarization expert. Your task is to create a concise summary of the following "
                    "conversation between a user and an AI assistant. Focus on the key information, decisions made, "
                    "code written or modified, and the overall progress of the task. If available, also mention relevant recent git commits.\n\n"
                    "## Conversation to Summarize\n\n"
                    f"{conversation_for_summary}"
                )

                if git_log_output:
                    summary_prompt += f"\n\n## Recent Git Commits\n\n{git_log_output}"

                summary_prompt += "\n\nYour summary should be self-contained and easy to understand for providing context in a new session."
                
                try:
                    with console.status("[bold yellow]Compressing conversation...[/]"):
                        response = litellm.completion(
                            model=model,
                            api_key=api_key,
                            messages=[{"role": "user", "content": summary_prompt}],
                            search=False, # Explicitly disable search for internal tasks
                        )
                        summary = response.choices[0].message.content
                except Exception as e:
                    console.print(f"[bold red]Error during compression:[/] {e}")
                    continue

                # Reset context
                messages = [{"role": "system", "content": get_system_prompt(agent_mode, cfg)}]
                read_files_in_session.clear()
                
                # Add summary to new context
                summary_user_prompt = (
                    "The previous conversation has been summarized to save context space. "
                    f"Here is the summary:\n\n{summary}\n\n"
                    "Based on this, please continue with your task or await further instructions."
                )
                messages.append({"role": "user", "content": summary_user_prompt})
                
                console.print(Panel(summary, title="[bold green]Conversation Summary[/]", border_style="green"))
                
                # Let the agent respond to the summary
                try:
                    process_llm_turn(messages, read_files_in_session, cfg, agent_mode, session_stats, yolo_mode=yolo_mode)
                except Exception as e:
                    console.print(f"[bold red]An error occurred after compression:[/] {e}")
                    messages.pop()

                continue
            elif user_input.lower().split()[0] == "/rag":
                parts = user_input.lower().split()
                command = parts[1] if len(parts) > 1 else None
                batch_size = cfg.get("rag_settings", {}).get("rag_batch_size", 100)

                if command == "init":
                    embedding_cfg = cfg.get("embedding")
                    if not embedding_cfg or not embedding_cfg.get("provider") or not embedding_cfg.get("model"):
                        console.print("[bold red]Error:[/] Embedding model not configured. Use `/config` to set it.")
                        continue
                    
                    api_key = _find_api_key_for_provider(cfg, embedding_cfg["provider"])
                    embedding_cfg["api_key"] = api_key
                    try:
                        rag_retriever = CodeRAG(
                            project_path=os.getcwd(),
                            config_dir=config.CONFIG_DIR,
                            embedding_config=embedding_cfg,
                            original_openai_api_base=original_openai_api_base
                        )
                        rag_retriever.index_project(batch_size=batch_size, force_reindex=False)
                        console.print("[bold green]RAG is now active.[/bold green]")
                    except Exception as e:
                        console.print(f"[bold red]Error initializing RAG:[/] {e}")
                        rag_retriever = None
                    continue
                
                elif command == "update":
                    if not rag_retriever:
                        console.print("[bold red]Error:[/] RAG is not initialized. Use `/rag init` first.")
                        continue
                    with console.status("[bold yellow]Re-indexing project for RAG...[/]"):
                        try:
                            rag_retriever.index_project(batch_size=batch_size, force_reindex=True)
                        except Exception as e:
                            console.print(f"[bold red]Error updating RAG index:[/] {e}")
                    continue

                elif command == "deinit":
                    rag_retriever = None
                    last_rag_context = ""
                    messages[0]['content'] = get_system_prompt(agent_mode, cfg)
                    console.print("[bold green]RAG has been deactivated for this session.[/bold green]")
                    continue
                
                else:
                    console.print("[bold red]Error:[/] Invalid RAG command. Use `/rag init`, `/rag update`, or `/rag deinit`.")
                    continue
            elif user_input.lower().split()[0] == "/load_preset":
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2:
                    presets = cfg.get("presets", {})
                    console.print("[bold red]Usage: /load_preset <preset_name>[/]")
                    if presets:
                        console.print(f"Available presets: {', '.join(sorted(presets.keys()))}")
                    else:
                        console.print("No presets saved.")
                    continue
                
                preset_name = parts[1]
                presets = cfg.get("presets", {})

                if preset_name not in presets:
                    console.print(f"[bold red]Error:[/] Preset '{preset_name}' not found.")
                    if presets:
                        console.print(f"Available presets: {', '.join(sorted(presets.keys()))}")
                    continue
                
                # Load the preset by updating the current config object
                preset_data = presets[preset_name]
                current_presets = cfg.get("presets", {}) # Persist the presets themselves
                cfg.clear()
                cfg.update(preset_data)
                cfg["presets"] = current_presets
                _update_environment_for_mode(agent_mode, cfg)
                
                # Reload system prompt as config has changed
                messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                
                console.print(f"[bold green]✔ Preset '{preset_name}' loaded. Agent capabilities updated.[/bold green]")
                continue
            elif user_input.lower().split()[0] == "/memory":
                parts = user_input.strip().split(maxsplit=2)
                command = parts[1] if len(parts) > 1 else None

                if command == "save":
                    if len(parts) < 3:
                        console.print("[bold red]Usage: /memory save <text to remember>[/]")
                        continue
                    text_to_save = parts[2]
                    result = tools.save_memory(text_to_save, scope="project", source="user")
                    console.print(f"[bold green]Memory saved:[/bold green] {result}")
                    if memories_active:
                        messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    continue
                
                elif command == "update":
                    if Confirm.ask("[bold yellow]This will delete all LLM-generated project memories and regenerate them. User-generated memories will be kept. Continue?[/]"):
                        run_in_background = cfg.get("memory_settings", {}).get("run_in_background", False)
                        _run_memory_update(cfg, memory_update_in_progress, run_in_background)
                    else:
                        console.print("[yellow]Memory update cancelled.[/yellow]")
                    continue

                elif command == "delete":
                    if Confirm.ask("[bold yellow]Are you sure you want to delete this project's memories? This cannot be undone.[/]"):
                        project_name = os.path.basename(os.getcwd())
                        project_memory_file = config.DATA_DIR / f"{project_name}.md"
                        
                        if project_memory_file.exists():
                            project_memory_file.unlink()
                            console.print("[bold green]Deleted project memory file.[/bold green]")
                        else:
                            console.print("[yellow]No project memory file to delete.[/yellow]")
                        
                        if memories_active:
                            messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    else:
                        console.print("[yellow]Memory deletion cancelled.[/yellow]")
                    continue

                elif command == "deinit":
                    memories_active = False
                    messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    console.print("[bold green]Memories unloaded from context for this session.[/bold green]")
                    continue

                elif command == "init":
                    memories_active = True
                    messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    console.print("[bold green]Memories loaded into context for this session.[/bold green]")
                    continue
                
                else:
                    console.print("[bold red]Invalid memory command. Usage: /memory <save|update|delete|init|deinit>[/]")
                    continue
            elif user_input.lower() == "/yolo":
                yolo_mode = not yolo_mode
                status = "[bold green]ON[/]" if yolo_mode else "[bold red]OFF[/]"
                console.print(f"👉 YOLO Mode is now {status}.")
                if yolo_mode:
                    console.print("[yellow]Warning: Dangerous commands will execute without confirmation.[/yellow]")
                continue
            elif user_input.lower().split()[0] == "/mode":
                parts = user_input.strip().lower().split()
                if len(parts) == 2 and parts[1] in MODES:
                    agent_mode = parts[1]
                    _update_environment_for_mode(agent_mode, cfg)
                    messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    console.print(f"Switched to [bold green]{agent_mode.capitalize()}[/bold green] mode.")
                elif len(parts) == 1 and parts[0] == "/mode":
                    console.print(f"Current mode: {agent_mode}. Available modes: {', '.join(MODES)}. Usage: /mode <mode_name>")
                else:
                    console.print(f"[red]Invalid mode or usage. Available modes: {', '.join(MODES)}[/red]")
                continue
            elif user_input.lower() in ["/exit", "exit"]:
                break
            elif user_input.lower().split()[0] == "/scaffold":
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) < 2 or not parts[1]:
                    console.print("[bold red]Usage: /scaffold <a description of the project to build>[/]")
                    continue
                
                scaffold_prompt = parts[1]

                # --- Step 1: Run Architect ---
                console.print(Panel(f"Running architect to design project structure for: '{scaffold_prompt}'", title="[bold blue]Scaffold Step 1: Architect[/]", border_style="blue"))
                architect_prompt = (
                    "Based on the following user request, create a detailed, step-by-step plan for a new software project. "
                    "Your plan should describe the directory structure and the purpose of each file to be created. Do NOT write implementation code. "
                    "Your final output MUST be just the plan, which will be passed to a coding agent.\n\n"
                    f"USER REQUEST: {scaffold_prompt}"
                )
                
                architect_result_json = run_sub_agent("architect", architect_prompt, cfg)
                try:
                    architect_result = json.loads(architect_result_json)
                    if architect_result.get("reason") != "success":
                        console.print(Panel(f"Architect agent failed.\nReason: {architect_result.get('reason')}\nInfo: {architect_result.get('info')}", title="[bold red]Scaffold Failed[/]", border_style="red"))
                        continue
                    
                    coding_plan = architect_result.get("info")
                    if not coding_plan:
                        console.print(Panel("Architect agent did not return a plan.", title="[bold red]Scaffold Failed[/]", border_style="red"))
                        continue
                except (json.JSONDecodeError, AttributeError) as e:
                    console.print(Panel(f"Could not parse architect agent output: {e}\nOutput: {architect_result_json}", title="[bold red]Scaffold Failed[/]", border_style="red"))
                    continue

                # --- Step 2: Run Coder ---
                console.print(Panel(coding_plan, title="[bold blue]Scaffold Step 2: Coder - Received Plan[/]", border_style="blue"))
                if not Confirm.ask("[bold yellow]The architect has created the above plan. Proceed with generating the code?[/]"):
                    console.print("[yellow]Scaffolding cancelled.[/yellow]")
                    continue

                code_prompt = (
                    "You are a coding agent tasked with scaffolding a new project. Based on the following plan from an architect, create the specified directories and files with the described content. "
                    "Make sure to create a complete, runnable starting point for the project.\n\n"
                    f"ARCHITECT'S PLAN:\n{coding_plan}"
                )
                
                code_result_json = run_sub_agent("code", code_prompt, cfg)
                try:
                    code_result = json.loads(code_result_json)
                    if code_result.get("reason") == "success":
                        console.print(Panel(f"Coding agent finished successfully.\nInfo: {code_result.get('info')}", title="[bold green]Scaffold Complete[/]", border_style="green"))
                    else:
                        console.print(Panel(f"Coding agent failed.\nReason: {code_result.get('reason')}\nInfo: {code_result.get('info')}", title="[bold red]Scaffold Failed[/]", border_style="red"))
                except (json.JSONDecodeError, AttributeError) as e:
                    console.print(Panel(f"Could not parse code agent output: {e}\nOutput: {code_result_json}", title="[bold red]Scaffold Failed[/]", border_style="red"))
                
                continue
            elif user_input.lower() == "/config":
                cfg = config.prompt_for_config()
                _update_environment_for_mode(agent_mode, cfg)
                # Reload system prompt in case settings affecting it (like tools) changed
                messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                continue
            elif user_input.lower().split()[0] == "/model":
                parts = user_input.strip().split()
                if len(parts) == 1:
                    # Show current model
                    modes = cfg.get("modes", {})
                    global_config = modes.get("global", {})
                    mode_config = modes.get(agent_mode, {})
                    active_provider = mode_config.get("active_provider") or global_config.get("active_provider")
                    
                    if active_provider:
                        global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
                        mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
                        final_provider_config = {**global_provider_settings, **mode_provider_settings}
                        model_name = final_provider_config.get("model")
                        if model_name:
                            console.print(f"Current model: {active_provider}/{model_name}")
                        else:
                            console.print("No model configured for current mode.")
                    else:
                        console.print("No provider configured for current mode.")
                    continue
                
                model_input = parts[1]
                if model_input.lower() in ["default", "reset"]:
                    # Reset to default model for the current mode
                    modes = cfg.get("modes", {})
                    global_config = modes.get("global", {})
                    mode_config = modes.get(agent_mode, {})
                    
                    # Remove any temporary model override
                    if "temp_model" in cfg:
                        del cfg["temp_model"]
                    
                    _update_environment_for_mode(agent_mode, cfg)
                    messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    console.print(f"[bold green]Model reset to default for {agent_mode} mode.[/bold green]")
                    continue
                
                # Set temporary model override
                if "/" not in model_input:
                    console.print("[bold red]Error:[/] Model must be in format 'provider/model_name'")
                    continue
                    
                provider, model_name = model_input.split("/", 1)
                if not provider or not model_name:
                    console.print("[bold red]Error:[/] Invalid model format. Use 'provider/model_name'")
                    continue
                
                # Store temporary model override
                cfg["temp_model"] = {
                    "provider": provider,
                    "model": model_name
                }
                
                _update_environment_for_mode(agent_mode, cfg)
                messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                console.print(f"[bold green]Model temporarily set to: {provider}/{model_name}[/bold green]")
                continue
            elif user_input.startswith('!'):
                command = user_input[1:].strip()
                if command:
                    title_command = command.splitlines()[0] if '\n' in command else command
                    output = tools.shell(command)
                    console.print(Panel(output, title=f"[bold yellow]! {title_command}[/]", border_style="yellow"))
                continue
                
            # Catch all other slash commands as unknown
            if user_input.startswith('/'):
                console.print(f"[bold red]Error:[/] Unknown command '{user_input.split()[0]}'. Type /help for a list of commands.")
                continue

            if rag_retriever:
                with console.status("[bold cyan]Searching RAG index...[/]"):
                    rag_context = rag_retriever.query(user_input)
                
                if rag_context and "No relevant context" not in rag_context:
                    last_rag_context = rag_context
                else:
                    last_rag_context = ""
            
            # Rebuild the system prompt to include the latest RAG/memory state
            messages[0]['content'] = get_system_prompt(agent_mode, cfg)

            messages.append({"role": "user", "content": user_input})
            try:
                process_llm_turn(messages, read_files_in_session, cfg, agent_mode, session_stats, yolo_mode=yolo_mode)
            except Exception as e:
                console.print(f"[bold red]An error occurred:[/] {e}")
                messages.pop()

            # --- RAG Auto-update logic ---
            if rag_retriever:
                rag_settings = cfg.get("rag_settings", {})
                auto_update_strategy = rag_settings.get("auto_update_strategy")
                run_in_background = rag_settings.get("run_in_background", False)

                update_triggered = False
                trigger_reason = ""
                
                if auto_update_strategy == "periodic":
                    prompt_count_since_rag_update += 1
                    prompt_interval = rag_settings.get("auto_update_prompt_interval", 5)
                    if prompt_count_since_rag_update >= prompt_interval:
                        # Only trigger update if there were edits
                        if session_stats.get("edit_count", 0) > 0:
                            update_triggered = True
                            trigger_reason = "prompt interval with edits"
                            prompt_count_since_rag_update = 0
                            session_stats["edit_count"] = 0
                
                elif auto_update_strategy == "edits":
                    edit_count = session_stats.get("edit_count", 0)
                    edit_interval = rag_settings.get("auto_update_edit_interval", 3)
                    if edit_count >= edit_interval:
                        update_triggered = True
                        trigger_reason = "edit interval"
                        session_stats["edit_count"] = 0
                
                if update_triggered:
                    console.print(f"[bold cyan]RAG auto-update triggered by {trigger_reason}.[/bold cyan]")
                    _run_rag_update(rag_retriever, rag_settings, rag_update_in_progress, run_in_background)

                elif auto_update_strategy == "model":
                    weak_model_cfg = cfg.get("weak_model", {})
                    provider = weak_model_cfg.get("provider")
                    model_name = weak_model_cfg.get("model")

                    if not provider or not model_name:
                        console.print("[bold yellow]Warning:[/] RAG auto-update by model is enabled, but no weak model is configured. Skipping.")
                    else:
                        # Check for uncommitted changes
                        git_diff_output = tools.shell("git diff HEAD")
                        
                        # Check if it's a git repo
                        is_git_repo = "Not a git repository" not in git_diff_output and "fatal:" not in git_diff_output
                        
                        if is_git_repo:
                            # Extract stdout and check if it's empty
                            stdout_match = re.search(r'STDOUT:\n(.*?)\nSTDERR:', git_diff_output, re.DOTALL)
                            has_changes = stdout_match and stdout_match.group(1).strip()
                            
                            if has_changes:
                                update_check_prompt = (
                                    "You are an AI assistant that determines if a codebase index needs to be updated based on file changes. "
                                    "Given the following `git diff`, respond with `<update>true</update>` if the changes are significant enough to warrant re-indexing the project for RAG, or `<update>false</update>` otherwise. "
                                    "Consider changes to source code, configurations, or documentation as significant. Ignore trivial changes like whitespace.\n\n"
                                    f"```diff\n{stdout_match.group(1)}\n```"
                                )
                                
                                try:
                                    with console.status("[bold yellow]Asking weak model about RAG update...[/]"):
                                        api_key = _find_api_key_for_provider(cfg, provider)
                                        model_to_use = f"{provider}/{model_name}"
                                        response = litellm.completion(
                                            model=model_to_use,
                                            api_key=api_key,
                                            messages=[{"role": "user", "content": update_check_prompt}],
                                            search=False, # Explicitly disable search for internal tasks
                                        )
                                        response_content = response.choices[0].message.content
                                        
                                    if "<update>true</update>" in response_content:
                                        console.print("[bold cyan]RAG auto-update triggered by weak model.[/bold cyan]")
                                        _run_rag_update(rag_retriever, rag_settings, rag_update_in_progress, run_in_background)

                                except Exception as e:
                                    console.print(f"[bold red]Error checking for RAG update with weak model:[/] {e}")

        except (KeyboardInterrupt, EOFError):
            break

        # --- Memory Auto-update logic ---
        memory_settings = cfg.get("memory_settings", {})
        auto_update_strategy = memory_settings.get("auto_update_strategy")
        
        if auto_update_strategy:
            run_in_background = memory_settings.get("run_in_background", False)
            update_triggered = False
            trigger_reason = ""

            if auto_update_strategy == "periodic":
                prompt_count_since_memory_update += 1
                prompt_interval = memory_settings.get("auto_update_prompt_interval", 5)
                if prompt_count_since_memory_update >= prompt_interval:
                    # Only trigger update if there were edits
                    if session_stats.get("edit_count", 0) > 0:
                        update_triggered = True
                        trigger_reason = "prompt interval with edits"
                        prompt_count_since_memory_update = 0
                        session_stats["edit_count"] = 0

            elif auto_update_strategy == "edits":
                edit_count = session_stats.get("edit_count", 0)
                edit_interval = memory_settings.get("auto_update_edit_interval", 3)
                if edit_count >= edit_interval:
                    update_triggered = True
                    trigger_reason = "edit interval"
                    session_stats["edit_count"] = 0
            
            if update_triggered:
                console.print(f"[bold cyan]Memory auto-update triggered by {trigger_reason}.[/bold cyan]")
                _run_memory_update(cfg, memory_update_in_progress, run_in_background)

            elif auto_update_strategy == "model":
                weak_model_cfg = cfg.get("weak_model", {})
                provider = weak_model_cfg.get("provider")
                model_name = weak_model_cfg.get("model")

                if not provider or not model_name:
                    console.print("[bold yellow]Warning:[/] Memory auto-update by model is enabled, but no weak model is configured. Skipping.")
                else:
                    git_diff_output = tools.shell("git diff HEAD")
                    is_git_repo = "Not a git repository" not in git_diff_output and "fatal:" not in git_diff_output
                    if is_git_repo:
                        stdout_match = re.search(r'STDOUT:\n(.*?)\nSTDERR:', git_diff_output, re.DOTALL)
                        has_changes = stdout_match and stdout_match.group(1).strip()
                        if has_changes:
                            update_check_prompt = (
                                "You are an AI assistant that determines if project memories need to be updated based on file changes. "
                                "Given the following `git diff`, respond with `<update>true</update>` if the changes are significant enough to warrant re-generating project memories, or `<update>false</update>` otherwise. "
                                "Consider changes to source code, configurations, or documentation as significant. Ignore trivial changes like whitespace.\n\n"
                                f"```diff\n{stdout_match.group(1)}\n```"
                            )
                            try:
                                with console.status("[bold yellow]Asking weak model about memory update...[/]"):
                                    api_key = _find_api_key_for_provider(cfg, provider)
                                    model_to_use = f"{provider}/{model_name}"
                                    response = litellm.completion(
                                        model=model_to_use,
                                        api_key=api_key,
                                        messages=[{"role": "user", "content": update_check_prompt}],
                                        search=False,
                                    )
                                    response_content = response.choices[0].message.content
                                
                                if "<update>true</update>" in response_content:
                                    console.print("[bold cyan]Memory auto-update triggered by weak model.[/bold cyan]")
                                    _run_memory_update(cfg, memory_update_in_progress, run_in_background)
                            except Exception as e:
                                console.print(f"[bold red]Error checking for memory update with weak model:[/] {e}")
    
    console.print("\n[bold yellow]Exiting interactive mode.[/]")
    # Restore original environment variable on exit
    if original_openai_api_base is not None:
        os.environ["OPENAI_API_BASE"] = original_openai_api_base
    elif "OPENAI_API_BASE" in os.environ:
        del os.environ["OPENAI_API_BASE"]

def main():
    """Main function for the agentic CLI tool."""
    # --- Main Parser ---
    parser = argparse.ArgumentParser(
        description="A command-line coding agent that uses LiteLLM."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Interactive session (default command)
    interactive_parser = subparsers.add_parser("interactive", help="Start an interactive session (default).")
    interactive_parser.add_argument(
        "prompt", type=str, nargs="?",
        default=sys.stdin.read() if not sys.stdin.isatty() else None,
        help="The initial prompt for the interactive session. Can be passed as an argument or piped via stdin.",
    )

    # Config command
    subparsers.add_parser("config", help="Open the configuration prompt.")

    # --- MCP Command Handling (as a subcommand of the main parser) ---
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Configure and manage MCP servers",
        description="Configure and manage MCP servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp_parser.usage = "agentic mcp [options] [command]"
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", title="Commands")

    # `mcp add` command
    parser_add = mcp_subparsers.add_parser(
        "add",
        help="Add a server"
    )
    parser_add.add_argument("name", help="A unique name for the server.")
    parser_add.add_argument("commandOrUrl", help="The command or URL of the server.")
    parser_add.add_argument("args", nargs=argparse.REMAINDER)
    parser_add.add_argument("--scope", choices=["user", "project", "local"], default="local", help="Where to save the configuration. 'user' is global, 'project' is for the repo, 'local' is for this machine only.")
    parser_add.add_argument("--header", action="append", help="A header to send with requests (e.g., 'X-API-Key:my-secret-key'). Can be specified multiple times.")

    # `mcp remove` command
    parser_remove = mcp_subparsers.add_parser(
        "remove",
        help="Remove an MCP server"
    )
    parser_remove.add_argument("name", help="The name of the server to remove.")
    parser_remove.add_argument("--scope", choices=["user", "project", "local"], help="The specific scope from which to remove the server.")

    # `mcp list` command
    mcp_subparsers.add_parser(
        "list",
        help="List configured MCP servers"
    )

    # `mcp get` command
    parser_get = mcp_subparsers.add_parser("get", help="Get details about an MCP server")
    parser_get.add_argument("name", help="The name of the server to get.")
    
    # `mcp add-json` command
    parser_add_json = mcp_subparsers.add_parser(
        "add-json",
        help="Add an MCP server (stdio or SSE) with a JSON string"
    )
    parser_add_json.add_argument("name", help="A unique name for the server.")
    parser_add_json.add_argument("json", help="The JSON configuration for the server.")
    parser_add_json.add_argument("--scope", choices=["user", "project", "local"], default="local", help="The scope to save the server to.")

    # `mcp add-from-claude-desktop` command
    mcp_subparsers.add_parser(
        "add-from-claude-desktop",
        help="Import MCP servers from Claude Desktop (Mac and WSL only)"
    )

    mcp_subparsers.add_parser("help", help="display help for command")

    # This is a bit of a hack to get the command arguments to show up in the help text for `agentic mcp`
    COMMAND_METAVARS = {
        'add': 'add [options] <name> <commandOrUrl> [args...]',
        'remove': 'remove [options] <name>',
        'get': 'get <name>',
        'add-json': 'add-json [options] <name> <json>',
        'add-from-claude-desktop': 'add-from-claude-desktop [options]',
        'help': 'help [command]'
    }
    if hasattr(mcp_subparsers, '_choices_actions'):
        for action in mcp_subparsers._choices_actions:
            if action.dest in COMMAND_METAVARS:
                action.metavar = COMMAND_METAVARS[action.dest]

    # Check if a command is provided, otherwise default to interactive
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in subparsers.choices):
        # Insert 'interactive' as the default command
        sys.argv.insert(1, 'interactive')

    args = parser.parse_args()
    
    if args.command == "mcp":
        if not args.mcp_command or args.mcp_command == "help":
            mcp_parser.print_help()
            sys.exit(0)

        if args.mcp_command == "add":
            headers = {k: v for k, v in (h.split(':', 1) for h in args.header)} if args.header else {}
            if args.args:
                server_config = {
                    "transport": "stdio",
                    "command": [args.commandOrUrl] + args.args
                }
            else:
                server_config = {
                    "transport": "http",
                    "url": args.commandOrUrl,
                    "headers": headers
                }
            result = mcp.save_mcp_server(args.name, server_config, args.scope)
            console.print(result)
        
        elif args.mcp_command == "list":
            servers = mcp.load_mcp_servers()
            if not servers:
                console.print("[yellow]No MCP servers configured in any scope.[/yellow]")
            else:
                console.print("[bold]MCP Servers (merged from all scopes):[/bold]")
                
                paths = mcp._get_config_paths()
                user_servers, project_servers, local_servers = {}, {}, {}
                try:
                    if paths['user'].exists() and paths['user'].stat().st_size > 0:
                        user_servers = json.load(open(paths['user'])).get('servers', {})
                    if paths['project'].exists() and paths['project'].stat().st_size > 0:
                        project_servers = json.load(open(paths['project'])).get('servers', {})
                    if paths['local'].exists() and paths['local'].stat().st_size > 0:
                        local_servers = json.load(open(paths['local'])).get('servers', {})
                except json.JSONDecodeError:
                    pass

                for name, server_details in sorted(servers.items()):
                    scope = "[dim]unknown[/dim]"
                    if name in local_servers:
                        scope = "[bold green]local[/bold green]"
                    elif name in project_servers:
                        scope = "[bold yellow]project[/bold yellow]"
                    elif name in user_servers:
                        scope = "[bold blue]user[/bold blue]"
                    
                    console.print(f"- [cyan]{name}[/cyan] ({scope}): {server_details.get('url')}")
        
        elif args.mcp_command == "remove":
            result = mcp.remove_mcp_server(args.name, args.scope)
            console.print(result)

        elif args.mcp_command == "get":
            servers = mcp.load_mcp_servers()
            if args.name not in servers:
                console.print(f"Error: Server '{args.name}' not found.")
            else:
                paths = mcp._get_config_paths()
                scope = "[dim]unknown[/dim]"
                try:
                    if paths['local'].exists() and paths['local'].stat().st_size > 0 and args.name in json.load(open(paths['local'])).get('servers', {}):
                        scope = "local"
                    elif paths['project'].exists() and paths['project'].stat().st_size > 0 and args.name in json.load(open(paths['project'])).get('servers', {}):
                        scope = "project"
                    elif paths['user'].exists() and paths['user'].stat().st_size > 0 and args.name in json.load(open(paths['user'])).get('servers', {}):
                        scope = "user"
                except json.JSONDecodeError:
                    pass
                
                console.print(f"[bold]Details for server '{args.name}' (from {scope} scope):[/bold]")
                console.print(json.dumps(servers[args.name], indent=2))
        
        elif args.mcp_command == "add-json":
            try:
                config_dict = json.loads(args.json)
                result = mcp.save_mcp_server(args.name, config_dict, args.scope)
                console.print(result)
            except json.JSONDecodeError:
                console.print("Error: Invalid JSON string provided.")
        
        elif args.mcp_command == "add-from-claude-desktop":
            mcp.copy_claude_code_mcp_config()
        
        sys.exit(0)

    cfg = config.load_config()

    if not cfg: # Truly empty config, first run
        console.print("[bold yellow]Welcome to Agentic! No configuration found, setting up with default (Hackclub AI).[/bold yellow]")
        try:
            HACKCLUB_API_BASE = "https://ai.hackclub.com"
            model_name = "AI Hackclub"

            default_cfg = {
                "modes": {
                    "global": {
                        "active_provider": "hackclub_ai",
                        "providers": {
                            "hackclub_ai": {
                                "model": model_name,
                            }
                        },
                        "tool_strategy": "xml",
                    }
                }
            }
            cfg = default_cfg
            config.save_config(cfg)
            console.print("[bold green]✔ Default configuration saved. Use `/config` to change it later.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] Could not set up default Hackclub AI config: {e}")
            console.print("Please configure manually.")
            # Fall through to manual configuration

    if not is_config_valid(cfg):
        console.print("[bold yellow]Your agent is not configured. Please set it up.[/bold yellow]")
        cfg = config.prompt_for_config()
        if not is_config_valid(cfg):
            console.print("[bold red]Active provider is not fully configured. Exiting.[/bold red]")
            sys.exit(1)

    if args.command == "config":
        config.prompt_for_config()
        sys.exit(0)

    # This handles the default 'interactive' command
    console.print(ASCII_LOGO)
    console.print(
        Panel(
            "Type '/help' for a list of commands.",
            subtitle="[cyan]Interactive Mode[/]",
            expand=False,
        )
    )
    start_interactive_session(args.prompt, cfg)

if __name__ == "__main__":
    main()
