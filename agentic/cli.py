#!/usr/bin/env python3

import argparse
import sys
import json
import os
import re
import xml.etree.ElementTree as ET
import litellm
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import is_done
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
from .tools import generate_xml_tool_prompt

console = Console()

class SubAgentEndTask(Exception):
    def __init__(self, reason: str, info: str = ""):
        self.reason = reason
        self.info = info or "" # Ensure info is a string
        super().__init__(f"Sub-agent ended task with reason: {reason}")

CODE_SYSTEM_PROMPT = (
    "You are an AI assistant expert in software development. You have access to a powerful set of tools.\n"
    "Your primary directive is to ALWAYS understand the project context before providing code or solutions.\n\n"
    "**Mandatory Workflow:**\n"
    "1. **Gather Context:** Start every task by using `ReadFolder` to see the project layout. Then, use `ReadFile` on the most relevant files to understand how the code works. Do not skip this step.\n"
    "2. **Think & Plan:** Use the `Think` tool to break down the problem, formulate a hypothesis, and create a step-by-step plan. This is a crucial step for complex tasks.\n"
    "3. **Ask for Feedback (if needed):** If the plan is complex or you are unsure about the best approach, use the `UserInput` tool to ask for clarification or confirmation before proceeding.\n"
    "4. **Analyze & Execute:** Based on your plan, use `SearchText`, `Edit`, `WriteFile`, or `Shell` to execute the steps.\n"
    "5. **Consult Web:** Use `WebFetch` if you need external information.\n"
    "6. **Finish:** Once the task is complete, call `EndTask` with a `reason` of 'success' and `info` summarizing your work. If you cannot complete the task, call `EndTask` with 'failure' and explain why.\n\n"
    "**Tool Guidelines:**\n"
    "- `Think`: Use this to externalize your thought process. It helps you structure your plan and analyze information before taking action.\n"
    "- `UserInput`: Use this to ask for feedback, clarification, or the next feature to implement. Essential for interactive development.\n"
    "- `WriteFile`: Creates a new file or completely overwrites an existing one. Use with caution.\n"
    "- `Edit`: Performs a targeted search-and-replace. This is safer for small changes.\n"
    "- `Shell`: Executes shell commands. Powerful but dangerous. Use it only when necessary.\n"
    "- `EndTask`: You MUST call this tool to signal you have finished your task. Provide a clear reason ('success' or 'failure') and a summary of your work in the 'info' field.\n\n"
    "Always use relative paths. Be methodical. Think step by step."
)

ASK_SYSTEM_PROMPT = (
    "You are an AI assistant designed to answer questions and provide information. You are in 'ask' mode.\n"
    "Your goal is to be a helpful and knowledgeable resource. You can read files and browse the web to find answers.\n\n"
    "**Workflow:**\n"
    "1. **Understand the Question:** Analyze the user's query to fully grasp what they are asking.\n"
    "2. **Gather Information:** Use `ReadFile` to examine relevant files and `WebFetch` to get external information if necessary.\n"
    "3. **Synthesize and Answer:** Combine the information you've gathered to provide a comprehensive and clear answer. Use `Think` to structure your thoughts.\n\n"
    "**Tool Guidelines:**\n"
    "- You do **not** have access to tools that modify files (`WriteFile`, `Edit`) or execute shell commands (`Shell`).\n"
    "- Focus on providing information and answering questions."
)

ARCHITECT_SYSTEM_PROMPT = (
    "You are a principal AI software architect. You are in 'architect' mode.\n"
    "Your purpose is to create high-level plans and designs for software projects. Do NOT write implementation code.\n\n"
    "**Workflow:**\n"
    "1. **Gather Context:** Use `ReadFolder` and `ReadFile` to understand the existing project structure and code.\n"
    "2. **Deconstruct the Request:** Use the `Think` tool to break down the user's request into architectural components and requirements.\n"
    "3. **Design the Plan:** Formulate a detailed, step-by-step plan. Describe new files to be created, changes to existing files, and the overall structure. Do not write the code for these changes.\n"
    "4. **Seek Clarification:** Use `UserInput` if the requirements are ambiguous or to get feedback on your proposed plan.\n"
    "5. **Finish:** When your plan is complete, call the `EndTask` tool. Set `reason` to 'success' and put your complete, final plan into the `info` parameter.\n\n"
    "**Tool Guidelines:**\n"
    "- Your primary tool is `Think` to outline your architectural plan.\n"
    "- You can read files, but you cannot write or edit them. You cannot use `Shell`.\n"
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


def _get_available_tools(agent_mode: str, is_sub_agent: bool) -> list:
    """Gets the list of available tools metadata based on the agent's mode."""
    disallowed_tools = set()

    # The 'memory' mode is the only one that should be able to save memories.
    if agent_mode != "memory":
        disallowed_tools.add("SaveMemory")

    if agent_mode in ["ask", "memory", "architect"]:
        disallowed_tools.update({"WriteFile", "Edit", "Shell"})
    elif agent_mode == "agent-maker":
        # Agent-maker can only read, think, and make sub-agents.
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

def run_sub_agent(mode: str, prompt: str, cfg: dict) -> str:
    """
    Runs a non-interactive sub-agent for a specific task.
    Returns a JSON string with the result from the sub-agent.
    """
    console.print(Panel(f"Starting sub-agent in '{mode}' mode...\nPrompt: {prompt}", title="[bold blue]Sub-agent Invoked[/]", border_style="blue"))

    # Determine tool strategy from config
    modes_cfg = cfg.get("modes", {})
    global_cfg = modes_cfg.get("global", {})
    mode_cfg = modes_cfg.get(mode, {})
    active_provider = mode_cfg.get("active_provider") or global_cfg.get("active_provider")
    provider_cfg = {**global_cfg.get("providers", {}).get(active_provider, {}), **mode_cfg.get("providers", {}).get(active_provider, {})}
    tool_strategy = provider_cfg.get("tool_strategy", "tool_calls")

    memories = load_memories()
    system_prompt_template = SYSTEM_PROMPTS.get(mode, CODE_SYSTEM_PROMPT)
    
    system_prompt_parts = []
    if tool_strategy == "xml":
        available_tools = _get_available_tools(mode, is_sub_agent=True)
        system_prompt_parts.append(generate_xml_tool_prompt(available_tools))

    if memories:
        system_prompt_parts.append(f"### PERMANENT MEMORIES ###\n{memories}")
    
    system_prompt_parts.append(f"### TASK ###\n{system_prompt_template}")
    system_prompt = "\n\n".join(filter(None, system_prompt_parts))

    sub_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    sub_read_files = set()

    try:
        # Run the agent loop. It will end by either returning a message (error)
        # or raising SubAgentEndTask (success/failure).
        final_message = process_llm_turn(sub_messages, sub_read_files, cfg, mode, yolo_mode=False, is_sub_agent=True)
        
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


def process_llm_turn(messages, read_files_in_session, cfg, agent_mode: str, yolo_mode: bool = False, is_sub_agent: bool = False):
    """Handles a single turn of the LLM, including tool calls and user confirmation."""
    DANGEROUS_TOOLS = {"WriteFile", "Edit", "Shell"}
    available_tools_metadata = _get_available_tools(agent_mode, is_sub_agent)

    # Get model and API key, falling back to global settings
    modes = cfg.get("modes", {})
    global_config = modes.get("global", {})
    mode_config = modes.get(agent_mode, {})

    active_provider = mode_config.get("active_provider") or global_config.get("active_provider")
    model, api_key = None, None
    tool_strategy = "tool_calls"

    if active_provider:
        # Mode-specific provider settings override global ones
        global_provider_settings = global_config.get("providers", {}).get(active_provider, {})
        mode_provider_settings = mode_config.get("providers", {}).get(active_provider, {})
        
        # Merge settings, with mode-specific taking precedence
        final_provider_config = {**global_provider_settings, **mode_provider_settings}
        
        model_name = final_provider_config.get("model")
        api_key = final_provider_config.get("api_key") # Can be None
        tool_strategy = final_provider_config.get("tool_strategy", "tool_calls")
        
        if model_name:
            model = f"{active_provider}/{model_name}"

    if not model: # Model is required, API key is not
        console.print(f"[bold red]Error:[/] Agent mode '{agent_mode}' is not configured (or is missing a model).")
        console.print("Please use `/config` to set the provider and model for this mode or for the 'global' settings.")
        return # Stop processing and return to the user prompt

    while True:
        if tool_strategy == 'tool_calls':
            response = litellm.completion(
                model=model,
                api_key=api_key,
                messages=messages,
                tools=available_tools_metadata,
                tool_choice="auto",
            )
            choice = response.choices[0]
            if choice.finish_reason != "tool_calls":
                break # Go to final streaming response
            
            messages.append(choice.message)
            tool_calls = choice.message.tool_calls
        else: # xml strategy
            response = litellm.completion(model=model, api_key=api_key, messages=messages)
            choice = response.choices[0]
            response_content = choice.message.content or ""
            
            messages.append(choice.message)
            
            tool_xml_blocks = re.findall(r'<tool_code>(.*?)</tool_code>', response_content, re.DOTALL)
            
            if not tool_xml_blocks:
                break # No tools, proceed to streaming response
            
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
        
        # --- Common Tool Execution Logic ---
        has_executed_tool = False
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

            # SPECIAL HANDLING FOR MakeSubagent
            if tool_name == "MakeSubagent":
                tool_output = run_sub_agent(
                    mode=tool_args["mode"],
                    prompt=tool_args["prompt"],
                    cfg=cfg
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id if is_native_call else tool_call["id"],
                    "name": tool_name,
                    "content": str(tool_output),
                })
                has_executed_tool = True
                continue

            if tool_name in DANGEROUS_TOOLS and not yolo_mode:
                if not Confirm.ask(
                    f"[bold yellow]Execute the [cyan]{tool_name}[/cyan] tool with the arguments above?[/]",
                    default=False
                ):
                    console.print("[bold red]Skipping tool call.[/]")
                    tool_output = "User denied execution of this tool call."
                else:
                    if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                        # Inject session-specific state if needed by the tool
                        if tool_name in ["ReadFile", "ReadManyFiles"]:
                            tool_args["read_files_in_session"] = read_files_in_session
                        with console.status("[bold yellow]Executing tool..."):
                            tool_output = tool_func(**tool_args)
                    else:
                        tool_output = f"Unknown tool '{tool_name}'"
            else:
                if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                    if tool_name in ["ReadFile", "ReadManyFiles"]:
                        tool_args["read_files_in_session"] = read_files_in_session
                    with console.status("[bold yellow]Executing tool..."):
                        tool_output = tool_func(**tool_args)
                else:
                    tool_output = f"Unknown tool '{tool_name}'"
            
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id if is_native_call else tool_call["id"],
                    "name": tool_name,
                    "content": str(tool_output),
                }
            )
            has_executed_tool = True

        if has_executed_tool:
            continue
        else: # No tools were run, break to final response
            break

    # This part handles the final text response from the assistant.
    if tool_strategy == 'xml' and messages[-1]['role'] == 'assistant':
        # For XML, we already have the complete final response. We just need to display it.
        full_response = messages[-1].get("content", "")
        # Remove any lingering tool code from the final output for cleaner display
        full_response = re.sub(r'<tool_code>.*?</tool_code>', '', full_response, flags=re.DOTALL).strip()
        
        panel = Panel(
            Markdown(full_response, style="default", code_theme="monokai"),
            title="[bold green]Assistant[/]",
            border_style="green",
        )
        console.print(panel)
    else:
        # For tool_calls, or if the XML flow produced no response, we stream.
        full_response = ""
        panel = Panel(
            "",
            title="[bold green]Assistant[/]",
            border_style="green",
        )
        with Live(panel, refresh_per_second=10, console=console) as live:
            # For tool_calls, we need a new completion call to get the final answer.
            # For XML, this path is hit if the LLM responds without tool calls,
            # so we need to stream out that response. We can't just call completion again.
            # However, the logic for XML now appends the message and breaks, so we should have it.
            # A second completion call is the original logic, let's stick to it for tool_calls.
            # The XML strategy's final response is handled above. This block is now tool_calls-centric.
            stream_response = litellm.completion(
                model=model, api_key=api_key, messages=messages, stream=True
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
        messages.append({"role": "assistant", "content": full_response})

    return messages[-1]

def display_help():
    """Displays the help menu for interactive commands."""
    help_text = """
| Command         | Description                                                 |
|-----------------|-------------------------------------------------------------|
| `/help`         | Show this help message.                                     |
| `/config`       | Open the configuration menu.                                |
| `/yolo`         | Toggle YOLO mode (disables safety confirmations).           |
| `/mode <name>`  | Switch agent mode (code, ask, architect).                   |
| `/exit` or `exit` | Exit the interactive session.                               |
| `! <command>`   | Execute a shell command directly from your terminal.        |

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

    def get_system_prompt(mode, current_cfg):
        # Determine tool strategy from config
        modes_cfg = current_cfg.get("modes", {})
        global_cfg = modes_cfg.get("global", {})
        mode_cfg = modes_cfg.get(mode, {})
        active_provider = mode_cfg.get("active_provider") or global_cfg.get("active_provider")
        provider_cfg = {**global_cfg.get("providers", {}).get(active_provider, {}), **mode_cfg.get("providers", {}).get(active_provider, {})}
        tool_strategy = provider_cfg.get("tool_strategy", "tool_calls")
        
        memories = load_memories()
        if memories and agent_mode == "code": # Only show memories loaded panel once at the start
            console.print(Panel(memories, title="[bold magenta]Memories Loaded[/]", border_style="magenta", expand=False))

        system_prompt_template = SYSTEM_PROMPTS[mode]
        
        system_prompt_parts = []
        if tool_strategy == "xml":
            available_tools = _get_available_tools(mode, is_sub_agent=False)
            system_prompt_parts.append(generate_xml_tool_prompt(available_tools))

        if memories:
            system_prompt_parts.append(f"### PERMANENT MEMORIES ###\n{memories}")
        
        system_prompt_parts.append(f"### TASK ###\n{system_prompt_template}")
        return "\n\n".join(filter(None, system_prompt_parts))

    messages = [{"role": "system", "content": get_system_prompt(agent_mode, cfg)}]
    read_files_in_session = set()
    history = InMemoryHistory()
    yolo_mode = False

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(Panel(initial_prompt, title="[bold blue]User[/]", border_style="blue"))
        messages.append({"role": "user", "content": initial_prompt})
        try:
            process_llm_turn(messages, read_files_in_session, cfg, agent_mode, yolo_mode=yolo_mode)
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
            
            input_frame = Frame(
                Window(
                    content=BufferControl(buffer=prompt_buffer),
                    get_line_prefix=get_line_prefix,
                    wrap_lines=True
                ),
                title=to_formatted_text(HTML("<b>Your Turn</b>")),
                style="fg:cyan"
            )

            toolbar_text = '<b>[Enter]</b> to send, <b>[Alt+Enter]</b> or <b>[Ctrl+Enter]</b> for new line, <b>/help</b> for commands.'
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
            elif user_input.lower() == "/yolo":
                yolo_mode = not yolo_mode
                status = "[bold green]ON[/]" if yolo_mode else "[bold red]OFF[/]"
                console.print(f"👉 YOLO Mode is now {status}.")
                if yolo_mode:
                    console.print("[yellow]Warning: Dangerous commands will execute without confirmation.[/yellow]")
                continue
            elif user_input.lower().startswith("/mode"):
                parts = user_input.strip().lower().split()
                if len(parts) == 2 and parts[1] in MODES:
                    agent_mode = parts[1]
                    messages[0] = {"role": "system", "content": get_system_prompt(agent_mode, cfg)}
                    console.print(f"Switched to [bold green]{agent_mode.capitalize()}[/bold green] mode.")
                elif len(parts) == 1 and parts[0] == "/mode":
                    console.print(f"Current mode: {agent_mode}. Available modes: {', '.join(MODES)}. Usage: /mode <mode_name>")
                else:
                    console.print(f"[red]Invalid mode or usage. Available modes: {', '.join(MODES)}[/red]")
                continue
            elif user_input.lower() in ["/exit", "exit"]:
                break
            elif user_input.lower() == "/config":
                cfg = config.prompt_for_config()
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

            messages.append({"role": "user", "content": user_input})
            try:
                process_llm_turn(messages, read_files_in_session, cfg, agent_mode, yolo_mode=yolo_mode)
            except Exception as e:
                console.print(f"[bold red]An error occurred:[/] {e}")
                messages.pop()

        except (KeyboardInterrupt, EOFError):
            break
    
    console.print("\n[bold yellow]Exiting interactive mode.[/]")

def main():
    """Main function for the agentic CLI tool."""
    cfg = config.load_config()

    if not is_config_valid(cfg):
        console.print("[bold yellow]Welcome to Agentic! Please configure your API key and model.[/]")
        cfg = config.prompt_for_config()
        if not is_config_valid(cfg):
            console.print("[bold red]Active provider is not fully configured. Exiting.[/]")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="A command-line coding agent that uses LiteLLM."
    )
    parser.add_argument(
        "prompt", type=str, nargs="?",
        default=sys.stdin.read() if not sys.stdin.isatty() else None,
        help="The initial prompt for the interactive session. Can be passed as an argument or piped via stdin.",
    )
    parser.add_argument(
        "--config", action="store_true", help="Open the configuration prompt."
    )
    args = parser.parse_args()

    if args.config:
        config.prompt_for_config()
        sys.exit(0)

    console.print(
        Panel(
            "Type '/help' for a list of commands.",
            title="[bold green]Agentic[/]",
            subtitle="[cyan]Interactive Mode[/]",
            expand=False,
        )
    )
    start_interactive_session(args.prompt, cfg)

if __name__ == "__main__":
    main()
