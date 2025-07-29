#!/usr/bin/env python3

import argparse
import sys
import json
import litellm
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from . import tools
from . import config

console = Console()

SYSTEM_PROMPT = (
    "You are an AI assistant that is an expert in writing and explaining code. "
    "You have access to a set of tools to interact with the file system and run commands. "
    "Use them when necessary to answer the user's request. When reading or editing files, "
    "use relative paths from the current working directory."
)

def process_llm_turn(messages, read_files_in_session, cfg):
    """Handles a single turn of the LLM, including tool calls."""
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
            
            with console.status("[bold yellow]Assistant is thinking..."):
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    console.print(
                        Panel(
                            f"[cyan]{tool_name}[/][default]({json.dumps(tool_args)})[/]",
                            title="[bold yellow]Tool Call[/]",
                            border_style="yellow",
                            expand=False,
                        )
                    )

                    if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                        if "read" in tool_name:
                            tool_args["read_files_in_session"] = read_files_in_session
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

def start_interactive_session(initial_prompt, cfg):
    """Runs the agent in interactive mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    read_files_in_session = set()
    session = PromptSession()

    # Define keybindings for multiline input
    bindings = KeyBindings()
    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    # Handle initial prompt if provided
    if initial_prompt:
        console.print(Panel(initial_prompt, title="[bold blue]User[/]", border_style="blue"))
        messages.append({"role": "user", "content": initial_prompt})
        try:
            process_llm_turn(messages, read_files_in_session, cfg)
        except Exception as e:
            console.print(f"[bold red]An error occurred:[/] {e}")
            messages.pop()

    while True:
        try:
            console.rule("[bold blue]Your Turn[/]", style="blue")
            user_input = session.prompt(
                HTML('<b>> </b>'),
                key_bindings=bindings,
                bottom_toolbar=HTML(
                    '<b>[Enter]</b> to send, <b>[Alt+Enter]</b> for new line, <b>!cmd</b> to run shell.'
                ),
            ).strip()

            if not user_input:
                continue

            if user_input.startswith('!'):
                command = user_input[1:].strip()
                if command:
                    output = tools.run_command(command)
                    console.print(Panel(output, title=f"[bold yellow]! {command}[/]", border_style="yellow"))
                continue
            elif user_input.lower() == "/config":
                cfg = config.prompt_for_config()
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                process_llm_turn(messages, read_files_in_session, cfg)
            except Exception as e:
                console.print(f"[bold red]An error occurred:[/] {e}")
                messages.pop()

        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Exiting interactive mode.[/]")
            break

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
            "Type '/config' to change settings, or 'exit' to end.",
            title="[bold green]Agentic[/]",
            subtitle="[cyan]Interactive Mode[/]",
            expand=False,
        )
    )
    start_interactive_session(args.prompt, cfg)

if __name__ == "__main__":
    main()
