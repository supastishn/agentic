#!/usr/bin/env python3

import argparse
import os
import sys
from dotenv import load_dotenv
import litellm

# System prompt to guide the AI
SYSTEM_PROMPT = "You are an AI assistant that is an expert in writing and explaining code."

def main():
    """
    Main function for the agentic CLI tool.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="A command-line coding agent that uses LiteLLM."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=sys.stdin.read() if not sys.stdin.isatty() else None,
        help="The coding prompt for the agent. Can be passed as an argument or piped via stdin.",
    )
    args = parser.parse_args()

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]

    try:
        response = litellm.completion(model="gpt-4o", messages=messages, stream=True)
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print() # for a final newline
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
