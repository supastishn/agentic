#!/usr/bin/env python3

import argparse
import sys
import json
import litellm
from . import tools
from . import config

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
            print("Assistant: Thinking...", flush=True)
            tool_calls = choice.message.tool_calls
            messages.append(choice.message)
            
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"Tool Call: {tool_name}({json.dumps(tool_args)})", flush=True)

                if tool_func := tools.AVAILABLE_TOOLS.get(tool_name):
                    if "read" in tool_name:
                        tool_args["read_files_in_session"] = read_files_in_session
                    tool_output = tool_func(**tool_args)
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "name": tool_name, "content": str(tool_output),
                    })
                else:
                    print(f"Warning: Unknown tool '{tool_name}' called.", file=sys.stderr)
            continue
        
        print("Assistant:")
        full_response = ""
        stream_response = litellm.completion(
            model=cfg["model"], api_key=cfg["api_key"], messages=messages, stream=True
        )
        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
        print()
        messages.append({"role": "assistant", "content": full_response})
        break

def start_interactive_session(initial_prompt, cfg):
    """Runs the agent in interactive mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    read_files_in_session = set()
    prompt = initial_prompt

    while True:
        if prompt:
            print(f"\nUser: {prompt}")
            messages.append({"role": "user", "content": prompt})
        
        try:
            process_llm_turn(messages, read_files_in_session, cfg)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            messages.pop()

        try:
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "/config":
                cfg = config.prompt_for_config()
                prompt = None
                continue
            prompt = user_input
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break

def main():
    """Main function for the agentic CLI tool."""
    cfg = config.load_config()
    if not cfg.get("api_key"):
        print("Welcome to Agentic! Please configure your API key and model.")
        cfg = config.prompt_for_config()
        if not cfg.get("api_key"):
            print("API key is required to run. Exiting.", file=sys.stderr)
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="A command-line coding agent that uses LiteLLM."
    )
    parser.add_argument(
        "prompt", type=str, nargs="?",
        default=sys.stdin.read() if not sys.stdin.isatty() else None,
        help="The prompt. Can be passed as an argument or piped via stdin.",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode."
    )
    parser.add_argument(
        "--config", action="store_true", help="Open the configuration prompt."
    )
    args = parser.parse_args()

    if args.config:
        config.prompt_for_config()
        sys.exit(0)

    if args.interactive:
        print("Entering interactive mode. Type '/config' to change settings, or 'exit' to end.")
        start_interactive_session(args.prompt, cfg)
    elif args.prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ]
        read_files_in_session = set()
        try:
            process_llm_turn(messages, read_files_in_session, cfg)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
