#!/usr/bin/env python3

import argparse
import os
import sys
from dotenv import load_dotenv
import litellm

# System prompt to guide the AI
SYSTEM_PROMPT = "You are an AI assistant that is an expert in writing and explaining code."

def call_llm_and_stream(messages):
    """
    Calls the LLM with the given messages, streams the response,
    and returns the full assistant message.
    """
    full_response = ""
    response = litellm.completion(model="gpt-4o", messages=messages, stream=True)
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content
    print()  # Final newline
    return full_response

def start_interactive_session(initial_prompt):
    """Runs the agent in interactive mode, maintaining conversation history."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt = initial_prompt

    while True:
        if prompt is None:
            try:
                prompt = input("\nUser: ")
                if prompt.lower() in ["exit", "quit"]:
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break
        else:
            # Print the initial prompt if it was passed as an argument
            print(f"\nUser: {prompt}")

        messages.append({"role": "user", "content": prompt})

        try:
            print("\nAssistant:")
            assistant_response = call_llm_and_stream(messages)
            messages.append({"role": "assistant", "content": assistant_response})
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            # Remove the last user message to allow retrying on the same prompt
            messages.pop()

        # Reset prompt for the next loop iteration
        prompt = None

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
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode.",
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
        try:
            call_llm_and_stream(messages)
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # No prompt and not interactive
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
