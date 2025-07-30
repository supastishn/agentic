#!/usr/bin/env python3

import argparse
import sys
import json
import litellm
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import is_done
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import Frame
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.formatted_text import to_formatted_text

from . import tools
from . import config

console = Console()

SYSTEM_PROMPT = (
    "You are an AI assistant expert in software development. You have access to a powerful set of tools.\n"
    "Your primary directive is to ALWAYS understand the project context before providing code or solutions.\n\n"
    "**Mandatory Workflow:**\n"
    "1. **Gather Context:** Start every task by using `ReadFolder` to see the project layout. Then, use `ReadFile` on the most relevant files to understand how the code works. Do not skip this step.\n"
    "2. **Think & Plan:** Use the `Think` tool to break down the problem, formulate a hypothesis, and create a step-by-step plan. This is a crucial step for complex tasks.\n"
    "3. **Ask for Feedback (if needed):** If the plan is complex or you are unsure about the best approach, use the `UserInput` tool to ask for clarification or confirmation before proceeding.\n"
    "4. **Analyze & Execute:** Based on your plan, use `SearchText`, `Edit`, `WriteFile`, or `Shell` to execute the steps. Use `SaveMemory` to remember key findings.\n"
    "5. **Consult Web:** Use `WebFetch` if you need external information.\n\n"
    "**Tool Guidelines:**\n"
    "- `Think`: Use this to externalize your thought process. It helps you structure your plan and analyze information before taking action.\n"
    "- `UserInput`: Use this to ask for feedback, clarification, or the next feature to implement. Essential for interactive development.\n"
    "- `WriteFile`: Creates a new file or completely overwrites an existing one. Use with caution.\n"
    "- `Edit`: Performs a targeted search-and-replace. This is safer for small changes.\n"
    "- `Shell`: Executes shell commands. Powerful but dangerous. Use it only when necessary.\n\n"
    "Always use relative paths. Be methodical. Think step by step."
)

def process_llm_turn(messages, read_files_in_session, cfg, yolo_mode: bool = False):
    """Handles a single turn of the LLM, including tool calls and user confirmation."""
    DANGEROUS_TOOLS = {"WriteFile", "Edit", "Shell"}

    while True:
        response = litellm.completion(
            model=cfg["model"],
            api_key=cfg["api_key"],
            messages=messages,
            tools=tools.TOOLS_METADATA,
            tool_choice="auto",
        )

        choice = response.choices[0]
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            tool_calls = choice.message.tool_calls
            
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_panel_content = f"[cyan]{tool_name}[/][default]({json.dumps(tool_args, indent=2)})[/]"
                console.print(
                    Panel(
                        tool_panel_content,
                        title="[bold yellow]Tool Call[/]",
                        border_style="yellow",
                        expand=False,
                    )
                )

                if tool_name in DANGEROUS_TOOLS and not yolo_mode:
                    if not Confirm.ask(
                        f"[bold yellow]Execute the [cyan]{tool_name}[/cyan] tool with the arguments above?[/]",
                        default=False
                    ):
                        console.print("[bold red]Skipping tool call.[/]")
                        # Provide feedback to the LLM that the user cancelled
                        tool_output = "User denied execution of this tool call."
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                        continue # Move to the next tool call or re-prompt

                if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                    # Inject session-specific state if needed by the tool
                    if tool_name in ["ReadFile", "ReadManyFiles"]:
                        tool_args["read_files_in_session"] = read_files_in_session
                    
                    with console.status("[bold yellow]Executing tool..."):
                        tool_output = tool_func(**tool_args)
                    
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(tool_output),
                        }
                    )
                else:
                    console.print(f"[bold red]Warning:[/] Unknown tool '{tool_name}' called.")
            continue
        
        full_response = ""
        panel = Panel(
            "",
            title="[bold green]Assistant[/]",
            border_style="green",
        )
        with Live(panel, refresh_per_second=10, console=console) as live:
            stream_response = litellm.completion(
                model=cfg["model"], api_key=cfg["api_key"], messages=messages, stream=True
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
        break

def display_help():
    """Displays the help menu for interactive commands."""
    help_text = """
| Command         | Description                                                 |
|-----------------|-------------------------------------------------------------|
| `/help`         | Show this help message.                                     |
| `/config`       | Open the configuration menu.                                |
| `/yolo`         | Toggle YOLO mode (disables safety confirmations).           |
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


def start_interactive_session(initial_prompt, cfg):
    """Runs the agent in interactive mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    read_files_in_session = set()
    yolo_mode = False

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(Panel(initial_prompt, title="[bold blue]User[/]", border_style="blue"))
        messages.append({"role": "user", "content": initial_prompt})
        try:
            process_llm_turn(messages, read_files_in_session, cfg, yolo_mode=yolo_mode)
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/] {e}")
            messages.pop()

    while True:
        try:
            # --- New prompt with a frame ---
            prompt_buffer = Buffer(multiline=True)

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
            
            messages.append({"role": "user", "content": user_input})
            try:
                process_llm_turn(messages, read_files_in_session, cfg, yolo_mode=yolo_mode)
            except Exception as e:
                console.print(f"[bold red]An error occurred:[/] {e}")
                messages.pop()

        except (KeyboardInterrupt, EOFError):
            break
    
    console.print("\n[bold yellow]Exiting interactive mode.[/]")

def main():
    """Main function for the agentic CLI tool."""
    cfg = config.load_config()
    if not cfg.get("api_key"):
        console.print("[bold yellow]Welcome to Agentic! Please configure your API key and model.[/]")
        cfg = config.prompt_for_config()
        if not cfg.get("api_key"):
            console.print("[bold red]API key is required to run. Exiting.[/]")
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
