import os
import json
from pathlib import Path
import sys
import litellm
import requests
from cryptography.fernet import Fernet
from rich.console import Console
from rich.panel import Panel
from simple_term_menu import TerminalMenu

def _get_provider_models() -> dict:
    """Gets available models from LiteLLM's model list JSON and groups them by provider."""
    MODELS_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
    provider_models = {}
    console = Console()

    try:
        response = requests.get(MODELS_URL, timeout=10)
        response.raise_for_status()
        model_data = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        console.print(f"[bold yellow]Warning:[/] Could not fetch model list from LiteLLM repo: {e}")
        return {}
    
    for model_key, model_info in model_data.items():
        provider = model_info.get("litellm_provider")
        if not provider:
            continue
        
        model_name = ""
        # If the model key from JSON is prefixed with the provider, strip it.
        # e.g., "replicate/..." -> "..."
        if model_key.startswith(f"{provider}/"):
            model_name = model_key.split("/", 1)[1]
        else:
            # Otherwise, use the key as the model name.
            # e.g., "gpt-4" -> "gpt-4"
            model_name = model_key
        
        if provider not in provider_models:
            provider_models[provider] = []
        
        if model_name:
            provider_models[provider].append(model_name)

    # Sort model names for consistent display
    for provider in provider_models:
        provider_models[provider].sort()
        
    return provider_models

# --- Constants ---
CONFIG_DIR = Path.home() / ".agentic-pypi"
CONFIG_FILE = CONFIG_DIR / "config.encrypted"
KEY_FILE = CONFIG_DIR / "config.key"

# --- Key Management ---

def _ensure_config_dir():
    """Ensures the configuration directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)

def _load_key() -> bytes:
    """Loads the encryption key, or generates it if it doesn't exist."""
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    
    _ensure_config_dir()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    # Set restrictive permissions for the key file
    os.chmod(KEY_FILE, 0o600)
    return key

# --- Configuration Load/Save ---

def load_config() -> dict:
    """Loads and decrypts the configuration from the config file."""
    if not CONFIG_FILE.exists():
        return {}
    
    key = _load_key()
    fernet = Fernet(key)
    
    try:
        encrypted_data = CONFIG_FILE.read_bytes()
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data)
    except Exception as e:
        print(f"Warning: Could not load configuration. It might be corrupted. {e}", file=sys.stderr)
        return {}

def save_config(config: dict):
    """Encrypts and saves the configuration to the config file."""
    _ensure_config_dir()
    key = _load_key()
    fernet = Fernet(key)
    
    config_data = json.dumps(config).encode("utf-8")
    encrypted_data = fernet.encrypt(config_data)
    
    CONFIG_FILE.write_bytes(encrypted_data)

def prompt_for_config() -> dict:
    """Interactively prompts the user for configuration settings using a menu."""
    console = Console()
    original_config = load_config()
    # Use a deep copy to allow for discarding changes
    config_to_edit = json.loads(json.dumps(original_config))

    provider_models = _get_provider_models()
    all_providers = sorted(list(provider_models.keys()))

    while True:
        console.clear()

        # Get current values for display
        active_provider = config_to_edit.get("active_provider")
        provider_config = {}
        if active_provider:
            providers = config_to_edit.setdefault("providers", {})
            provider_config = providers.setdefault(active_provider, {})

        model = provider_config.get("model", "Not set")
        api_key = provider_config.get("api_key")
        api_key_display = f"****{api_key[-4:]}" if api_key else "Not set"

        config_view_content = (
            f"[bold cyan]Active Provider:[/bold cyan] {active_provider or 'Not set'}\n"
            f"[bold cyan]Model:[/bold cyan] {model}\n"
            f"[bold cyan]API Key:[/bold cyan] {api_key_display}"
        )
        console.print(Panel(config_view_content, title="[bold green]Current Configuration[/]", expand=False))

        menu_items = [
            "1. Select Provider",
            "2. Edit Model",
            "3. Edit API Key",
            None,  # Use None for a separator
            "4. Save and Exit",
            "5. Exit without Saving",
        ]

        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select. (ESC to discard)",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
            cycle_cursor=True,
            clear_screen=False,  # We handle clearing
        )

        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 4:  # Exit without Saving or ESC
            console.print("\n[yellow]Configuration changes discarded.[/yellow]")
            return original_config

        if selected_index == 0:  # Select Provider
            if not all_providers:
                console.print("\n[yellow]Could not dynamically determine providers.[/yellow]")
                console.input("Press Enter to continue...")
                continue

            provider_menu = TerminalMenu(all_providers, title="Select a provider")
            selected_provider_index = provider_menu.show()
            if selected_provider_index is not None:
                config_to_edit["active_provider"] = all_providers[
                    selected_provider_index
                ]

        elif selected_index == 1:  # Edit Model
            active_provider = config_to_edit.get("active_provider")
            if not active_provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue

            models = provider_models.get(active_provider, [])
            if not models:
                console.print(
                    f"\n[yellow]No models found for '{active_provider}'. You can enter one manually.[/yellow]"
                )
                new_model = console.input(
                    f"Enter model for {active_provider}: "
                ).strip()
            else:
                model_menu = TerminalMenu(
                    models, title=f"Select a model for {active_provider}"
                )
                selected_model_index = model_menu.show()
                new_model = (
                    models[selected_model_index]
                    if selected_model_index is not None
                    else None
                )

            if new_model:
                providers = config_to_edit.setdefault("providers", {})
                provider_cfg = providers.setdefault(active_provider, {})
                provider_cfg["model"] = new_model

        elif selected_index == 2:  # Edit API Key
            active_provider = config_to_edit.get("active_provider")
            if not active_provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue

            new_api_key = console.input(
                f"Enter new API Key for {active_provider}: "
            ).strip()
            if new_api_key:
                providers = config_to_edit.setdefault("providers", {})
                provider_cfg = providers.setdefault(active_provider, {})
                provider_cfg["api_key"] = new_api_key

        elif selected_index == 3:  # Save and Exit
            active_provider = config_to_edit.get("active_provider")
            if not active_provider:
                console.print(
                    "\n[bold red]An active provider must be selected. Configuration not saved.[/bold red]"
                )
                console.input("Press Enter to continue...")
                continue

            provider_config = config_to_edit.get("providers", {}).get(
                active_provider, {}
            )
            if not provider_config.get("model") or not provider_config.get("api_key"):
                console.print(
                    f"[bold red]Model and API Key are required for provider '{active_provider}'. Configuration not saved.[/bold red]"
                )
                console.input("Press Enter to continue...")
                continue

            save_config(config_to_edit)
            console.print("\n[bold green]✔ Configuration saved successfully.[/bold green]")
            return config_to_edit
