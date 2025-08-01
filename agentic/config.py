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
    """Gets available chat models from LiteLLM JSON, caches them, and groups them by provider."""
    MODELS_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
    console = Console()
    model_data = {}

    # 1. Try to fetch from URL and update cache
    try:
        response = requests.get(MODELS_URL, timeout=10)
        response.raise_for_status()
        model_data = response.json()
        
        _ensure_config_dir()
        with open(MODELS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(model_data, f)
    except requests.RequestException as e:
        console.print(f"[bold yellow]Warning:[/] Could not fetch model list from LiteLLM repo: {e}")
        # 2. If fetch fails, try to load from cache
        if MODELS_CACHE_FILE.exists():
            console.print(f"Attempting to use cached model list from '{MODELS_CACHE_FILE}'.")
            try:
                with open(MODELS_CACHE_FILE, "r", encoding="utf-8") as f:
                    model_data = json.load(f)
            except (json.JSONDecodeError, IOError) as cache_e:
                console.print(f"[bold red]Error:[/] Could not read or parse model cache file: {cache_e}")
                return {} # Failed to load cache, so we can't proceed
        else:
            console.print("[bold red]Error:[/] No cached model list found. Connect to the internet to download it.")
            return {} # No internet and no cache, so we can't proceed

    # 3. Parse the loaded model data
    provider_models = {}
    for model_key, model_info in model_data.items():
        # Filter out non-dictionary entries and specific keys like 'litellm_spec'
        if not isinstance(model_info, dict) or model_key == "litellm_spec":
            continue

        # Filter for chat models only
        if model_info.get("mode") != "chat":
            continue

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
DATA_DIR = CONFIG_DIR / "data"
CONFIG_FILE = CONFIG_DIR / "config.encrypted"
KEY_FILE = CONFIG_DIR / "config.key"
MODELS_CACHE_FILE = CONFIG_DIR / "model_cache.json"

# --- Key Management ---

def _ensure_config_dir():
    """Ensures the configuration directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)

def _ensure_data_dir():
    """Ensures the data directory for memories exists."""
    DATA_DIR.mkdir(exist_ok=True)

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

def _prompt_for_one_mode(config_to_edit: dict, mode_name: str, provider_models: dict, all_providers: list):
    """Interactively prompts for a single mode's configuration."""
    console = Console()
    
    # Work on a specific slice of the config
    mode_cfg = config_to_edit["modes"].setdefault(mode_name, {})
    # Keep a backup to revert if user cancels
    original_mode_cfg = json.loads(json.dumps(mode_cfg))

    while True:
        console.clear()

        active_provider = mode_cfg.get("active_provider")
        provider_config = {}
        if active_provider:
            providers = mode_cfg.setdefault("providers", {})
            provider_config = providers.setdefault(active_provider, {})

        model = provider_config.get("model", "Not set")
        api_key = provider_config.get("api_key")
        api_key_display = f"****{api_key[-4:]}" if api_key else "Not set (Optional)"

        config_view_content = (
            f"[bold cyan]Provider:[/bold cyan] {active_provider or 'Not set'}\n"
            f"[bold cyan]Model:[/bold cyan] {model}\n"
            f"[bold cyan]API Key:[/bold cyan] {api_key_display}"
        )
        console.print(Panel(config_view_content, title=f"[bold green]Configuring '{mode_name.capitalize()}' Mode[/]", expand=False))

        menu_items = [
            "1. Select Provider",
            "2. Edit Model",
            "3. Edit API Key",
            None,
        ]
        # Dynamically build menu
        next_option_num = 4
        reset_option_idx, save_option_idx, discard_option_idx = None, None, None

        if mode_name != "global":
            menu_items.append(f"{next_option_num}. Reset to Global Defaults")
            reset_option_idx = len(menu_items) - 1
            next_option_num += 1
        
        menu_items.append(None)

        save_option_idx = len(menu_items)
        menu_items.append(f"{next_option_num}. Back (Save Changes)")
        next_option_num += 1

        discard_option_idx = len(menu_items)
        menu_items.append(f"{next_option_num}. Back (Discard Changes)")

        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select.",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
        )
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == discard_option_idx:  # Discard and Back
            config_to_edit["modes"][mode_name] = original_mode_cfg
            break
        
        if selected_index == reset_option_idx: # Reset to Global
            if mode_name in config_to_edit["modes"]:
                del config_to_edit["modes"][mode_name]
            break

        if selected_index == 0:  # Select Provider
            if not all_providers:
                console.print("\n[yellow]Could not determine providers.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            provider_menu = TerminalMenu(all_providers, title="Select a provider")
            sel_provider_idx = provider_menu.show()
            if sel_provider_idx is not None:
                new_provider = all_providers[sel_provider_idx]
                # If provider changes, preserve model/key if they exist under the new provider
                existing_settings = mode_cfg.get("providers", {}).get(new_provider, {})
                mode_cfg["active_provider"] = new_provider
                mode_cfg.setdefault("providers", {})[new_provider] = existing_settings
            continue

        elif selected_index == 1:  # Edit Model
            active_provider = mode_cfg.get("active_provider")
            if not active_provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            models = provider_models.get(active_provider, [])
            if not models:
                console.print(f"\n[yellow]No models found for '{active_provider}'. Enter one manually.[/yellow]")
                new_model = console.input(f"Enter model for {active_provider}: ").strip()
            else:
                model_menu = TerminalMenu(models, title=f"Select a model for {active_provider}")
                sel_model_idx = model_menu.show()
                new_model = models[sel_model_idx] if sel_model_idx is not None else None
            
            if new_model:
                mode_cfg["providers"].setdefault(active_provider, {})["model"] = new_model
            continue

        elif selected_index == 2:  # Edit API Key
            active_provider = mode_cfg.get("active_provider")
            if not active_provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            new_api_key = console.input(f"Enter new API Key for {active_provider} (optional, press Enter to clear): ").strip()
            if new_api_key:
                mode_cfg["providers"].setdefault(active_provider, {})["api_key"] = new_api_key
            else:
                mode_cfg.get("providers", {}).get(active_provider, {}).pop("api_key", None)
            continue

        elif selected_index == save_option_idx:  # Save and Back
            active_provider = mode_cfg.get("active_provider")
            if not active_provider:
                # This case means the user cleared the provider. We should remove the mode config.
                if mode_name in config_to_edit["modes"]:
                    del config_to_edit["modes"][mode_name]
                break

            provider_cfg = mode_cfg.get("providers", {}).get(active_provider, {})
            # Model is required if a provider is specified. API key is not.
            if not provider_cfg.get("model"):
                console.print(f"[bold red]A Model must be specified for provider '{active_provider}'. Changes not saved.[/bold red]")
                console.input("Press Enter to continue...")
                continue
            
            # Changes are already in config_to_edit, so we just break
            break


def prompt_for_config() -> dict:
    """Interactively prompts the user to select a mode and then configure it."""
    console = Console()
    original_config = load_config()
    config_to_edit = json.loads(json.dumps(original_config)) # Deep copy

    provider_models = _get_provider_models()
    all_providers = sorted(list(provider_models.keys()))
    # Add 'global' and 'agent-maker' modes
    all_modes = ["global", "code", "ask", "architect", "agent-maker", "memory"]
    config_to_edit.setdefault("modes", {})

    while True:
        console.clear()
        console.print(Panel("Select a mode to configure, or save/exit.", title="[bold green]Configuration Menu[/]", expand=False))

        modes_cfg = config_to_edit.get("modes", {})
        global_cfg = modes_cfg.get("global", {})

        menu_items = []
        for mode_name in all_modes:
            mode_config = modes_cfg.get(mode_name, {})
            
            # Determine provider and model, with fallback to global for non-global modes
            provider = mode_config.get("active_provider")
            model = ""
            source = ""

            if provider:
                model = mode_config.get("providers", {}).get(provider, {}).get("model", "")
            elif mode_name != "global" and global_cfg.get("active_provider"):
                provider = global_cfg.get("active_provider")
                model = global_cfg.get("providers", {}).get(provider, {}).get("model", "")
                if model:
                    source = " (uses Global)"

            display_model = f"{provider}/{model}{source}" if provider and model else "Not Configured"
            menu_items.append(f"{mode_name.capitalize():<15} ({display_model})")
        
        menu_items.extend([None, "Save and Exit", "Exit without Saving"])
        
        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select.",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
        )
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == len(all_modes) + 2: # "Exit without Saving" or ESC
            console.print("\n[yellow]Configuration changes discarded.[/yellow]")
            return original_config
        
        if selected_index == len(all_modes) + 1: # "Save and Exit"
            # Remove empty mode configurations before saving
            modes_to_del = [m for m, c in config_to_edit.get("modes", {}).items() if not c]
            for m in modes_to_del:
                if m in config_to_edit["modes"]:
                    del config_to_edit["modes"][m]

            save_config(config_to_edit)
            console.print("\n[bold green]✔ Configuration saved successfully.[/bold green]")
            return config_to_edit
        
        if selected_index is not None and selected_index < len(all_modes):
            selected_mode = all_modes[selected_index]
            _prompt_for_one_mode(config_to_edit, selected_mode, provider_models, all_providers)
            # The loop will now continue, re-rendering the main menu
