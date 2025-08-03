# agentic-cli-coder

An open-source, powerful, and versatile coding agent for your terminal. agentic-cli-coder is designed to be your primary AI assistant for software development, providing a rich set of tools and features to accelerate your workflow right from the command line.

![Agentic Demo](https://user-images.githubusercontent.com/12345/placeholder.gif) <!-- Placeholder: Replace with an actual demo GIF -->

## ✨ Features

*   **Multi-Provider LLM Support**: Powered by [LiteLLM](https://github.com/BerriAI/litellm), Agentic supports over 100 LLM providers, including OpenAI, Anthropic, Google, Mistral, and many more. You can even use local models via Ollama.
*   **Multiple Agent Modes**: Switch between specialized agents tailored for different tasks:
    *   `code`: A general-purpose coding assistant with a full toolset.
    *   `ask`: A question-answering agent with web search and file reading capabilities.
    *   `architect`: A high-level planner that designs project structures and implementation plans.
    *   `agent-maker`: A master agent that can delegate tasks to other sub-agents.
    *   `memory`: An agent dedicated to creating and managing project summaries.
*   **Retrieval-Augmented Generation (RAG)**: Agentic can index your entire project codebase to provide contextually-aware answers and code modifications.
*   **Persistent Memory**: Save key information, architectural summaries, and user instructions to a persistent memory that is loaded in future sessions.
*   **Powerful Interactive Shell**: A feature-rich interactive prompt with commands for managing conversation, configuration, and agent state.
*   **Sub-Agent Delegation**: Break down complex tasks by creating and delegating work to specialized sub-agents.
*   **Comprehensive Toolset**: Comes with a wide range of tools for file operations (`ReadFile`, `WriteFile`, `Edit`), version control (`Git`), shell command execution (`Shell`), web browsing, and more.
*   **Model Context Protocol (MCP) Integration**: Connect to external MCP servers to extend the agent's capabilities with third-party tools.
*   **Secure Configuration**: All sensitive information, including API keys, is encrypted on your local machine.

## 🚀 Installation

agentic-cli-coder is available on PyPI.

1.  **Install the package:**
    ```bash
    pip install agentic-cli-coder
    ```

2.  **Install browser drivers for the `Browser` tool:**
    Agentic uses Playwright for web browsing tasks. You need to install its browser drivers.
    ```bash
    playwright install
    ```

## 🏃‍♀️ Quick Start

1.  **Launch the agent:**
    ```bash
    agentic
    ```

2.  **Configure your LLM provider:**
    The first time you run Agentic, you will be prompted to configure it. You can also run the configuration menu at any time with the `/config` command.
    ```
    > /config
    ```
    This will open an interactive menu where you can select your provider (e.g., `openai`), enter your model name (e.g., `gpt-4-turbo`), and add your API key.

3.  **Start coding!**
    Once configured, you can start interacting with the agent.
    ```
    > Please write a Python script to list all files in the current directory.
    ```

## 📖 Usage

### Interactive Commands

Agentic provides a set of slash commands to manage the session:

| Command                  | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `/help`                  | Show the help message with all commands.                         |
| `/config`                | Open the interactive configuration menu.                         |
| `/mode <name>`           | Switch agent mode (e.g., `/mode ask`).                           |
| `/clear`                 | Clear the current conversation history.                          |
| `/rag <init|update>`     | Initialize or update the RAG index for the project.              |
| `/memory <init|save>`    | Load memories into context or save new information.              |
| `/yolo`                  | Toggle YOLO mode to execute dangerous tools without confirmation.|
| `/exit`                  | Exit the interactive session.                                    |
| `! <command>`            | Execute a shell command (e.g., `!ls -l`).                        |

### Non-Interactive Usage

You can pipe content directly into Agentic for non-interactive tasks.

```bash
cat my_file.py | agentic "Refactor this code to be more idiomatic."
```

## ⚙️ Configuration

agentic-cli-coder stores its configuration in `~/.agentic-cli-coder/`.
- `config.encrypted`: Encrypted file containing your settings and API keys.
- `config.key`: The local encryption key for your configuration.
- `data/`: Directory for persistent memories and RAG indexes.

The easiest way to manage your settings is through the interactive `agentic config` or `/config` command. You can set up different models for different modes, manage presets, configure RAG and memory settings, and much more.

## ❤️ Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.
