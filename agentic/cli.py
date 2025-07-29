#!/usr/bin/env python3

import argparse
import os
import sys
import json
from dotenv import load_dotenv
import litellm
from . import tools

SYSTEM_PROMPT = (
    "You are an AI assistant that is an expert in writing and explaining code. "
    "You have access to a set of tools to interact with the file system and run commands. "
    "Use them when necessary to answer the user's request. When reading or editing files, "
    "use relative paths from the current working directory."
)

def process_llm_turn(messages, read_files_in_session):
    """Handles a single turn of the LLM, including tool calls."""
    while True:
        response = litellm.completion(
            model="gpt-4o",
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
                    # Pass the set of read files to the read tools
                    if "read" in tool_name:
                        tool_args["read_files_in_session"] = read_files_in_session
                    
                    tool_output = tool_func(**tool_args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(tool_output),
                    })
                else:
                    print(f"Warning: Unknown tool '{tool_name}' called.", file=sys.stderr)
            # Continue loop to send tool results back to LLM
            continue
        
        # If no tool calls, stream the final response
        print("Assistant:")
        full_response = ""
        stream_response = litellm.completion(model="gpt-4o", messages=messages, stream=True)
        for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_response += content
        print() # Final newline
        messages.append({"role": "assistant", "content": full_response})
        break # Exit loop after streaming response

def start_interactive_session(initial_prompt):
    """Runs the agent in interactive mode."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    read_files_in_session = set() # Tracks files read in this session
    prompt = initial_prompt

    while True:
        if prompt:
            print(f"\nUser: {prompt}")
            messages.append({"role": "user", "content": prompt})
        
        try:
            process_llm_turn(messages, read_files_in_session)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            messages.pop() # Remove the last user message on error

        # Get next prompt from user
        try:
            prompt = input("\nUser: ")
            if prompt.lower() in ["exit", "quit"]:
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break

def main():
    """Main function for the agentic CLI tool."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="A command-line coding agent that uses LiteLLM."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=sys.stdin.read() if not sys.stdin.isatty() else None,
        help="The coding prompt. Can be passed as an argument or piped via stdin.",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive mode."
    )
    args = parser.parse_args()

    if args.interactive:
        print("Entering interactive mode. Type 'exit' or 'quit' to end.")
        start_interactive_session(args.prompt)
    elif args.prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.prompt},
        ]
        read_files_in_session = set()
        try:
            process_llm_turn(messages, read_files_in_session)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
