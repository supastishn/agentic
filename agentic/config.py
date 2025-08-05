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
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from . import mcp # Import the mcp module

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
    provider_models_info = {}
    for model_key, model_info in model_data.items():
        # Filter out non-dictionary entries and specific keys like 'litellm_spec'
        if not isinstance(model_info, dict) or model_key == "litellm_spec":
            continue

        # Filter for chat or embedding models only
        model_mode = model_info.get("mode")
        if model_mode not in ["chat", "embedding"]:
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
        
        if provider not in provider_models_info:
            provider_models_info[provider] = {}
        
        if model_name:
            # Default to False for function calling, and True for system messages if not specified.
            supports_function_calling = model_info.get("supports_function_calling", False)
            supports_system_message = model_info.get("supports_system_message", True)
            supports_web_search = model_info.get("supports_web_search", False)
            max_input_tokens = model_info.get("max_input_tokens", None)
            input_cost_per_token = model_info.get("input_cost_per_token", None)
            output_cost_per_token = model_info.get("output_cost_per_token", None)

            provider_models_info[provider][model_name] = {
                "supports_function_calling": supports_function_calling,
                "supports_system_message": supports_system_message,
                "supports_web_search": supports_web_search,
                "max_input_tokens": max_input_tokens,
                "input_cost_per_token": input_cost_per_token,
                "output_cost_per_token": output_cost_per_token,
                "mode": model_mode,
            }

    # Sort model names for consistent display
    for provider in provider_models_info:
        sorted_models = sorted(provider_models_info[provider].items())
        provider_models_info[provider] = dict(sorted_models)
        
    return provider_models_info

# --- Constants ---
CONFIG_DIR = Path.home() / ".agentic-cli-coder"
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

def _prompt_for_save_preset(config_to_edit: dict):
    """Saves the current configuration as a named preset."""
    console = Console()
    console.clear()
    
    preset_name = console.input("Enter a name for this preset: ").strip()
    if not preset_name:
        console.print("\n[yellow]Preset name cannot be empty.[/yellow]")
        console.input("Press Enter to continue...")
        return

    # The config to save is everything EXCEPT the presets themselves
    config_to_save = {k: v for k, v in config_to_edit.items() if k != "presets"}
    
    presets = config_to_edit.setdefault("presets", {})
    if preset_name in presets:
        if TerminalMenu(["Yes", "No"], title=f"Preset '{preset_name}' already exists. Overwrite?").show() != 0:
            console.print("\n[yellow]Save cancelled.[/yellow]")
            console.input("Press Enter to continue...")
            return
    
    presets[preset_name] = config_to_save
    console.print(f"\n[bold green]✔ Preset '{preset_name}' saved.[/bold green]")
    console.input("Press Enter to continue...")

def _prompt_for_load_preset(config_to_edit: dict):
    """Loads a configuration from a saved preset."""
    console = Console()
    presets = config_to_edit.get("presets", {})
    if not presets:
        console.print("\n[yellow]No presets have been saved.[/yellow]")
        console.input("Press Enter to continue...")
        return
        
    preset_names = sorted(list(presets.keys()))
    terminal_menu = TerminalMenu(preset_names, title="Select a preset to load")
    selected_index = terminal_menu.show()
    
    if selected_index is not None:
        selected_preset_name = preset_names[selected_index]
        preset_data = presets[selected_preset_name]
        
        # Clear the current config, but keep the presets dict
        current_presets = config_to_edit.get("presets", {})
        config_to_edit.clear()
        config_to_edit.update(preset_data)
        config_to_edit["presets"] = current_presets # Restore presets
        
        console.print(f"\n[bold green]✔ Preset '{selected_preset_name}' loaded.[/bold green]")
        console.input("Press Enter to continue...")

def _prompt_for_delete_preset(config_to_edit: dict):
    """Deletes a saved preset."""
    console = Console()
    presets = config_to_edit.get("presets", {})
    if not presets:
        console.print("\n[yellow]No presets to delete.[/yellow]")
        console.input("Press Enter to continue...")
        return
        
    preset_names = sorted(list(presets.keys()))
    terminal_menu = TerminalMenu(preset_names, title="Select a preset to delete")
    selected_index = terminal_menu.show()

    if selected_index is not None:
        selected_preset_name = preset_names[selected_index]
        
        if TerminalMenu(["Yes", "No"], title=f"Are you sure you want to delete preset '{selected_preset_name}'?").show() == 0:
            del config_to_edit["presets"][selected_preset_name]
            console.print(f"\n[bold green]✔ Preset '{selected_preset_name}' deleted.[/bold green]")
        else:
            console.print("\n[yellow]Deletion cancelled.[/yellow]")
            
        console.input("Press Enter to continue...")

def _prompt_for_weak_model_config(config_to_edit: dict, provider_models: dict, all_providers: list):
    """Interactively prompts for weak model configuration."""
    console = Console()

    # Work on a specific slice of the config
    weak_model_cfg = config_to_edit.setdefault("weak_model", {})
    # Keep a backup to revert if user cancels
    original_weak_model_cfg = json.loads(json.dumps(weak_model_cfg))

    while True:
        console.clear()

        provider = weak_model_cfg.get("provider")
        model = weak_model_cfg.get("model", "Not set")

        config_view_content = (
            f"[bold cyan]Provider:[/bold cyan] {provider or 'Not set'}\n"
            f"[bold cyan]Model:[/bold cyan] {model}"
        )
        console.print(Panel(config_view_content, title="[bold green]Configuring Weak Model[/]", expand=False))

        menu_items = [
            "1. Select Provider",
            "2. Edit Model",
            None,
            "3. Back (Save Changes)",
            "4. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select.",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
        )
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 4:  # Discard and Back
            config_to_edit["weak_model"] = original_weak_model_cfg
            # Ensure key is removed if it was empty before
            if not config_to_edit["weak_model"]:
                config_to_edit.pop("weak_model", None)
            break

        if selected_index == 0:  # Select Provider
            CUSTOM_PROVIDER_OPTION = "Custom..."
            provider_menu_items = [CUSTOM_PROVIDER_OPTION] + all_providers
            provider_menu = TerminalMenu(provider_menu_items, title="Select a provider")
            sel_provider_idx = provider_menu.show()

            if sel_provider_idx is None:
                continue

            new_provider = None
            if sel_provider_idx == 0:
                new_provider = console.input("Enter custom provider name: ").strip()
            else:
                new_provider = provider_menu_items[sel_provider_idx]

            if new_provider:
                if weak_model_cfg.get("provider") != new_provider:
                    weak_model_cfg.pop("model", None)
                weak_model_cfg["provider"] = new_provider
            continue

        elif selected_index == 1:  # Edit Model
            provider = weak_model_cfg.get("provider")
            if not provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue

            CUSTOM_MODEL_OPTION = "Custom..."
            models_for_provider = list(provider_models.get(provider, {}).keys())
            menu_items = [CUSTOM_MODEL_OPTION] + models_for_provider

            model_menu = TerminalMenu(menu_items, title=f"Select a model for {provider}")
            sel_model_idx = model_menu.show()

            if sel_model_idx is None:
                continue

            new_model = None
            if sel_model_idx == 0: # Custom selected
                new_model = console.input(f"Enter custom model name for {provider}: ").strip()
            else:
                new_model = menu_items[sel_model_idx]

            if new_model:
                weak_model_cfg["model"] = new_model
            continue

        elif selected_index == 3:  # Save and Back
            # If config is incomplete, remove it to keep config clean
            if not weak_model_cfg.get("provider") or not weak_model_cfg.get("model"):
                config_to_edit.pop("weak_model", None)
            break

def _prompt_for_embedding_config(config_to_edit: dict, provider_models: dict):
    """Interactively prompts for embedding model configuration."""
    console = Console()
    
    # Filter providers to only those that have at least one embedding model
    all_providers = sorted([
        p for p, models in provider_models.items()
        if any(m.get("mode") == "embedding" for m in models.values())
    ])

    # Work on a specific slice of the config
    emb_cfg = config_to_edit.setdefault("embedding", {})
    # Keep a backup to revert if user cancels
    original_emb_cfg = json.loads(json.dumps(emb_cfg))

    while True:
        console.clear()

        provider = emb_cfg.get("provider")
        model = emb_cfg.get("model", "Not set")

        config_view_content = (
            f"[bold cyan]Provider:[/bold cyan] {provider or 'Not set'}\n"
            f"[bold cyan]Model:[/bold cyan] {model}"
        )
        console.print(Panel(config_view_content, title="[bold green]Configuring Embedding Model[/]", expand=False))

        menu_items = [
            "1. Select Provider",
            "2. Edit Model",
            None,
            "3. Back (Save Changes)",
            "4. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select.",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
        )
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 4:  # Discard and Back
            config_to_edit["embedding"] = original_emb_cfg
            if not config_to_edit["embedding"]:
                config_to_edit.pop("embedding", None)
            break
        
        if selected_index == 0:  # Select Provider
            CUSTOM_PROVIDER_OPTION = "Custom..."
            provider_menu_items = [CUSTOM_PROVIDER_OPTION] + all_providers
            provider_menu = TerminalMenu(provider_menu_items, title="Select a provider")
            sel_provider_idx = provider_menu.show()

            if sel_provider_idx is None:
                continue
            
            new_provider = None
            if sel_provider_idx == 0:
                new_provider = console.input("Enter custom provider name: ").strip()
            else:
                new_provider = provider_menu_items[sel_provider_idx]

            if new_provider:
                if emb_cfg.get("provider") != new_provider:
                    emb_cfg.pop("model", None)
                emb_cfg["provider"] = new_provider
            continue

        elif selected_index == 1:  # Edit Model
            provider = emb_cfg.get("provider")
            if not provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            
            CUSTOM_MODEL_OPTION = "Custom..."
            models_for_provider = [
                name for name, info in provider_models.get(provider, {}).items()
                if info.get("mode") == "embedding"
            ]
            menu_items = [CUSTOM_MODEL_OPTION] + models_for_provider

            model_menu = TerminalMenu(menu_items, title=f"Select an embedding model for {provider}")
            sel_model_idx = model_menu.show()

            if sel_model_idx is None:
                continue
            
            new_model = None
            if sel_model_idx == 0: # Custom selected
                new_model = console.input(f"Enter custom embedding model for {provider}: ").strip()
            else:
                new_model = menu_items[sel_model_idx]
            
            if new_model:
                emb_cfg["model"] = new_model
            continue

        elif selected_index == 3:  # Save and Back
            if not emb_cfg.get("provider") or not emb_cfg.get("model"):
                config_to_edit.pop("embedding", None)
            break


def _prompt_for_one_mode(config_to_edit: dict, mode_name: str, provider_models: dict, all_providers: list):
    """Interactively prompts for a single mode's configuration."""
    console = Console()
    
    HACKCLUB_AI_KEY = "hackclub_ai"
    HACKCLUB_AI_DISPLAY_NAME = "Hackclub AI (No setup needed!)"
    HACKCLUB_API_BASE = "https://ai.hackclub.com"

    # Work on a specific slice of the config
    mode_cfg = config_to_edit["modes"].setdefault(mode_name, {})
    # Keep a backup to revert if user cancels
    original_mode_cfg = json.loads(json.dumps(mode_cfg))

    while True:
        console.clear()

        active_provider = mode_cfg.get("active_provider")
        is_hackclub = active_provider == HACKCLUB_AI_KEY
        
        provider_config = {}
        if active_provider:
            providers = mode_cfg.setdefault("providers", {})
            provider_config = providers.setdefault(active_provider, {})

        model = provider_config.get("model", "Not set")
        api_key = provider_config.get("api_key")
        api_key_display = "Not required" if is_hackclub else (f"****{api_key[-4:]}" if api_key else "Not set (Optional)")
        tool_strategy = mode_cfg.get("tool_strategy", "xml" if is_hackclub else "tool_calls")
        enable_web_search = provider_config.get("enable_web_search", False)

        model_capabilities = provider_models.get(active_provider, {}).get(model, {})
        supports_web_search = model_capabilities.get("supports_web_search", False)
        supports_tool_calls = model_capabilities.get("supports_function_calling", False)

        model_details = []
        if model != "Not set":
            fc = "✅" if supports_tool_calls else "❌"
            ws = "✅" if supports_web_search else "❌"
            model_details.append(f"Tools: {fc}, Search: {ws}")

            ctx = model_capabilities.get("max_input_tokens")
            if ctx:
                model_details.append(f"Context: {ctx:,}")
            
            in_cost = model_capabilities.get("input_cost_per_token")
            out_cost = model_capabilities.get("output_cost_per_token")
            if in_cost is not None and out_cost is not None:
                model_details.append(f"Cost/Mtk: ${in_cost * 1_000_000:.2f} (in), ${out_cost * 1_000_000:.2f} (out)")

        model_display_str = f"{model} ({' | '.join(model_details)})" if model_details else model

        config_view_content = (
            f"[bold cyan]Provider:[/bold cyan] {HACKCLUB_AI_DISPLAY_NAME if is_hackclub else active_provider or 'Not set'}\n"
            f"[bold cyan]Model:[/bold cyan] {model_display_str}\n"
            f"[bold cyan]API Key:[/bold cyan] {api_key_display}\n"
            f"[bold cyan]Tool Strategy:[/bold cyan] {tool_strategy}"
        )
        if supports_web_search:
            config_view_content += f"\n[bold cyan]Web Search:[/bold cyan] {'On' if enable_web_search else 'Off'}"
        console.print(Panel(config_view_content, title=f"[bold green]Configuring '{mode_name.capitalize()}' Mode[/]", expand=False))

        menu_items = [
            "1. Select Provider",
            "2. Edit Model" if not is_hackclub else "[dim]2. Edit Model (N/A)[/dim]",
            "3. Edit API Key" if not is_hackclub else "[dim]3. Edit API Key (N/A)[/dim]",
            "4. Edit Tool Strategy" if not is_hackclub else "[dim]4. Edit Tool Strategy (N/A)[/dim]",
        ]

        web_search_option_idx = -1
        if supports_web_search:
            web_search_option_idx = len(menu_items)
            menu_items.append(f"5. Toggle Web Search")

        menu_items.append(None)
        # Dynamically build menu
        next_option_num = 6
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

        if selected_index == web_search_option_idx:
            if supports_web_search:
                provider_config["enable_web_search"] = not enable_web_search
            continue
        
        if selected_index == reset_option_idx: # Reset to Global
            if mode_name in config_to_edit["modes"]:
                del config_to_edit["modes"][mode_name]
            break

        if selected_index == 0:  # Select Provider
            CUSTOM_PROVIDER_OPTION = "Custom..."
            provider_menu_items = [HACKCLUB_AI_DISPLAY_NAME, CUSTOM_PROVIDER_OPTION] + all_providers
            provider_menu = TerminalMenu(provider_menu_items, title="Select a provider")
            sel_provider_idx = provider_menu.show()

            if sel_provider_idx is None:
                continue

            new_provider_key = None
            if sel_provider_idx == 0: # Hackclub AI selected
                new_provider_key = HACKCLUB_AI_KEY
                model_name = "AI Hackclub"
                
                mode_cfg["active_provider"] = new_provider_key
                providers = mode_cfg.setdefault("providers", {})
                providers[new_provider_key] = {
                    "model": model_name,
                }
                # Force tool strategy and remove API key
                mode_cfg["tool_strategy"] = "xml"
                providers[new_provider_key].pop("api_key", None)
            
            elif provider_menu_items[sel_provider_idx] == CUSTOM_PROVIDER_OPTION:
                custom_provider = console.input("Enter custom provider name: ").strip()
                if custom_provider:
                    new_provider_key = custom_provider
                else:
                    continue # No input, do nothing

            else: # Other provider selected
                new_provider_key = provider_menu_items[sel_provider_idx]

            if new_provider_key and new_provider_key != HACKCLUB_AI_KEY:
                existing_settings = mode_cfg.get("providers", {}).get(new_provider_key, {})
                mode_cfg["active_provider"] = new_provider_key
                mode_cfg.setdefault("providers", {})[new_provider_key] = existing_settings
            
            continue

        elif is_hackclub and selected_index in [1, 2, 3]:
            console.print("\n[yellow]This setting cannot be changed for Hackclub AI.[/yellow]")
            console.input("Press Enter to continue...")
            continue
        
        elif selected_index == 1:  # Edit Model
            active_provider = mode_cfg.get("active_provider")
            if not active_provider:
                console.print("\n[yellow]Please select a provider first.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            
            CUSTOM_MODEL_OPTION = "Custom..."
            models_for_provider_dict = provider_models.get(active_provider, {})
            model_names = list(models_for_provider_dict.keys())
            
            menu_entries = [CUSTOM_MODEL_OPTION]
            for name in model_names:
                caps = models_for_provider_dict.get(name, {})
                fc = "✅" if caps.get("supports_function_calling") else "❌"
                ws = "✅" if caps.get("supports_web_search") else "❌"
                
                ctx = caps.get("max_input_tokens")
                ctx_str = f"Ctx: {ctx // 1000}k" if ctx else ""

                in_cost = caps.get("input_cost_per_token")
                out_cost = caps.get("output_cost_per_token")
                cost_str = ""
                if in_cost is not None and out_cost is not None:
                    in_mtk = in_cost * 1_000_000
                    out_mtk = out_cost * 1_000_000
                    cost_str = f"In: ${in_mtk:.2f} Out: ${out_mtk:.2f}"
                
                details = ", ".join(filter(None, [ctx_str, cost_str]))
                
                menu_entries.append(f"{name} (Tools: {fc}, Search: {ws}, {details})")
            
            model_menu = TerminalMenu(menu_entries, title=f"Select a model for {active_provider}")
            sel_model_idx = model_menu.show()
            
            if sel_model_idx is None:
                continue

            new_model = None
            if sel_model_idx == 0: # Custom selected
                custom_model = console.input(f"Enter custom model name for {active_provider}: ").strip()
                if custom_model:
                    new_model = custom_model
            else:
                # Extract model name from "model_name (details...)"
                new_model = menu_entries[sel_model_idx].split(" (")[0]
            
            if new_model:
                mode_cfg["providers"].setdefault(active_provider, {})["model"] = new_model
                # If model doesn't support tool_calls, force strategy to xml
                # For custom models, we assume they don't support tool_calls unless we know otherwise.
                model_capabilities = models_for_provider_dict.get(new_model, {})
                supports_fc = model_capabilities.get("supports_function_calling", False)
                if not supports_fc:
                    mode_cfg["tool_strategy"] = "xml"
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

        elif selected_index == 3:  # Edit Tool Strategy
            model = provider_config.get("model")
            
            supports_fc = False
            if active_provider and model:
                model_capabilities = provider_models.get(active_provider, {}).get(model, {})
                supports_fc = model_capabilities.get("supports_function_calling", False)

            if not supports_fc:
                console.print("\n[yellow]The selected model does not support native tool calls. Tool strategy is locked to 'xml'.[/yellow]")
                console.input("Press Enter to continue...")
                continue
            
            strategies = ["tool_calls", "xml"]
            strategy_menu = TerminalMenu(
                strategies,
                title="Select a tool strategy",
                menu_cursor="> ",
                menu_cursor_style=("fg_green", "bold"),
                menu_highlight_style=("bg_green", "fg_black"),
            )
            sel_strategy_idx = strategy_menu.show()
            if sel_strategy_idx is not None:
                mode_cfg["tool_strategy"] = strategies[sel_strategy_idx]
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


def _prompt_for_rag_auto_update(settings: dict):
    """Sub-menu for configuring RAG auto-update strategy."""
    console = Console()
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        strategy = settings.get("auto_update_strategy", "manual")
        
        strategy_display = {
            "manual": "Manual (only with /rag update)",
            "periodic": f"Every {settings.get('auto_update_prompt_interval', 'N/A')} prompts",
            "edits": f"Every {settings.get('auto_update_edit_interval', 'N/A')} edits",
            "model": "Use weak model to decide"
        }

        config_view_content = f"[bold cyan]Auto-update Strategy:[/bold cyan] {strategy_display.get(strategy)}"
        console.print(Panel(config_view_content, title="[bold green]RAG Auto-Update Settings[/]", expand=False))

        menu_items = [
            "1. Manual (only with /rag update)",
            "2. Every X prompts",
            "3. Every X edits",
            "4. Use weak model to decide",
            None,
            "5. Back (Save Changes)",
            "6. Back (Discard Changes)",
        ]
        
        terminal_menu = TerminalMenu(menu_items, title="Select a strategy", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 5: # Discard
            settings.clear()
            settings.update(original_settings)
            break
        
        if selected_index == 0:
            settings["auto_update_strategy"] = "manual"
            settings.pop("auto_update_prompt_interval", None)
            settings.pop("auto_update_edit_interval", None)
        elif selected_index == 1:
            settings["auto_update_strategy"] = "periodic"
            settings.pop("auto_update_edit_interval", None)
            interval_str = console.input("Enter prompt interval (e.g., 5): ").strip()
            try:
                interval = int(interval_str)
                if interval <= 0: raise ValueError
                settings["auto_update_prompt_interval"] = interval
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a positive whole number.[/bold red]")
                console.input("Press Enter to continue...")
        elif selected_index == 2:
            settings["auto_update_strategy"] = "edits"
            settings.pop("auto_update_prompt_interval", None)
            interval_str = console.input("Enter edit interval (e.g., 3): ").strip()
            try:
                interval = int(interval_str)
                if interval <= 0: raise ValueError
                settings["auto_update_edit_interval"] = interval
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a positive whole number.[/bold red]")
                console.input("Press Enter to continue...")
        elif selected_index == 3:
            settings["auto_update_strategy"] = "model"
            settings.pop("auto_update_prompt_interval", None)
            settings.pop("auto_update_edit_interval", None)
        elif selected_index == 4: # Save
            break

def _prompt_for_rag_settings(config_to_edit: dict):
    """Interactively prompts for RAG settings."""
    console = Console()
    
    settings = config_to_edit.setdefault("rag_settings", {})
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()

        auto_init = settings.get("auto_init_rag", False)
        batch_size = settings.get("rag_batch_size", 100)
        strategy = settings.get("auto_update_strategy", "manual")
        run_in_background = settings.get("run_in_background", False)

        strategy_display = {
            "manual": "Manual",
            "periodic": f"Periodic ({settings.get('auto_update_prompt_interval', 'N/A')} prompts)",
            "edits": f"Edits-based ({settings.get('auto_update_edit_interval', 'N/A')} edits)",
            "model": "Model-based"
        }

        config_view_content = (
            f"[bold cyan]Auto-initialize RAG on startup:[/bold cyan] {'On' if auto_init else 'Off'}\n"
            f"[bold cyan]Indexing Batch Size:[/bold cyan] {batch_size}\n"
            f"[bold cyan]Auto-update Strategy:[/bold cyan] {strategy_display.get(strategy, 'Manual')}\n"
            f"[bold cyan]Run Auto-updates in background:[/bold cyan] {'On' if run_in_background else 'Off'}"
        )
        console.print(Panel(config_view_content, title="[bold green]RAG Settings[/]", expand=False))

        menu_items = [
            f"1. Toggle Auto-init (current: {'On' if auto_init else 'Off'})",
            "2. Edit Batch Size",
            "3. Configure Auto-update Strategy",
            f"4. Toggle background updates (current: {'On' if run_in_background else 'Off'})",
            None,
            "5. Back (Save Changes)",
            "6. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 5:
            config_to_edit["rag_settings"] = original_settings
            if not config_to_edit["rag_settings"]:
                config_to_edit.pop("rag_settings", None)
            break
        
        if selected_index == 0:
            settings["auto_init_rag"] = not auto_init
            continue
        elif selected_index == 1:
            new_size_str = console.input("Enter new batch size (e.g., 100): ").strip()
            try:
                new_size = int(new_size_str)
                if new_size <= 0: raise ValueError
                settings["rag_batch_size"] = new_size
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a positive whole number.[/bold red]")
                console.input("Press Enter to continue...")
            continue
        elif selected_index == 2:
            _prompt_for_rag_auto_update(settings)
            continue
        elif selected_index == 3:
            settings["run_in_background"] = not run_in_background
            continue
        elif selected_index == 4:
            break

def _prompt_for_memory_auto_update(settings: dict):
    """Sub-menu for configuring Memory auto-update strategy."""
    console = Console()
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        strategy = settings.get("auto_update_strategy", "manual")
        
        strategy_display = {
            "manual": "Manual (only with /memory update)",
            "periodic": f"Every {settings.get('auto_update_prompt_interval', 'N/A')} prompts",
            "edits": f"Every {settings.get('auto_update_edit_interval', 'N/A')} edits",
            "model": "Use weak model to decide"
        }

        config_view_content = f"[bold cyan]Auto-update Strategy:[/bold cyan] {strategy_display.get(strategy)}"
        console.print(Panel(config_view_content, title="[bold green]Memory Auto-Update Settings[/]", expand=False))

        menu_items = [
            "1. Manual (only with /memory update)",
            "2. Every X prompts",
            "3. Every X edits",
            "4. Use weak model to decide",
            None,
            "5. Back (Save Changes)",
            "6. Back (Discard Changes)",
        ]
        
        terminal_menu = TerminalMenu(menu_items, title="Select a strategy", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 5: # Discard
            settings.clear()
            settings.update(original_settings)
            break
        
        if selected_index == 0:
            settings["auto_update_strategy"] = "manual"
            settings.pop("auto_update_prompt_interval", None)
            settings.pop("auto_update_edit_interval", None)
        elif selected_index == 1:
            settings["auto_update_strategy"] = "periodic"
            settings.pop("auto_update_edit_interval", None)
            interval_str = console.input("Enter prompt interval (e.g., 5): ").strip()
            try:
                interval = int(interval_str)
                if interval <= 0: raise ValueError
                settings["auto_update_prompt_interval"] = interval
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a positive whole number.[/bold red]")
                console.input("Press Enter to continue...")
        elif selected_index == 2:
            settings["auto_update_strategy"] = "edits"
            settings.pop("auto_update_prompt_interval", None)
            interval_str = console.input("Enter edit interval (e.g., 3): ").strip()
            try:
                interval = int(interval_str)
                if interval <= 0: raise ValueError
                settings["auto_update_edit_interval"] = interval
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a positive whole number.[/bold red]")
                console.input("Press Enter to continue...")
        elif selected_index == 3:
            settings["auto_update_strategy"] = "model"
            settings.pop("auto_update_prompt_interval", None)
            settings.pop("auto_update_edit_interval", None)
        elif selected_index == 4: # Save
            break

def _prompt_for_memory_settings(config_to_edit: dict):
    """Interactively prompts for Memory settings."""
    console = Console()
    
    settings = config_to_edit.setdefault("memory_settings", {})
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        auto_init = settings.get("auto_init_memories", False)
        strategy = settings.get("auto_update_strategy", "manual")
        run_in_background = settings.get("run_in_background", False)

        strategy_display = {
            "manual": "Manual",
            "periodic": f"Periodic ({settings.get('auto_update_prompt_interval', 'N/A')} prompts)",
            "edits": f"Edits-based ({settings.get('auto_update_edit_interval', 'N/A')} edits)",
            "model": "Model-based"
        }

        config_view_content = (
            f"[bold cyan]Auto-initialize Memories on startup:[/bold cyan] {'On' if auto_init else 'Off'}\n"
            f"[bold cyan]Auto-update Strategy:[/bold cyan] {strategy_display.get(strategy, 'Manual')}\n"
            f"[bold cyan]Run Auto-updates in background:[/bold cyan] {'On' if run_in_background else 'Off'}"
        )
        console.print(Panel(config_view_content, title="[bold green]Memory Settings[/]", expand=False))

        menu_items = [
            f"1. Toggle Auto-init (current: {'On' if auto_init else 'Off'})",
            "2. Configure Auto-update Strategy",
            f"3. Toggle background updates (current: {'On' if run_in_background else 'Off'})",
            None,
            "4. Back (Save Changes)",
            "5. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 4:
            config_to_edit["memory_settings"] = original_settings
            if not config_to_edit["memory_settings"]:
                config_to_edit.pop("memory_settings", None)
            break
        elif selected_index == 0:
            settings["auto_init_memories"] = not auto_init
            continue
        elif selected_index == 1:
            _prompt_for_memory_auto_update(settings)
            continue
        elif selected_index == 2:
            settings["run_in_background"] = not run_in_background
            continue
        elif selected_index == 3:
            break

def _prompt_for_tools_settings(config_to_edit: dict):
    """Interactively prompts for Tools settings."""
    console = Console()
    
    settings = config_to_edit.setdefault("tools_settings", {})
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        enable_user_input = settings.get("enable_user_input", False)
        enable_think = settings.get("enable_think", True)

        config_view_content = (
            f"[bold cyan]Enable UserInput Tool:[/bold cyan] {'On' if enable_user_input else 'Off'}\n"
            f"[bold cyan]Enable Think Tool:[/bold cyan] {'On' if enable_think else 'Off'}"
        )
        console.print(Panel(config_view_content, title="[bold green]Tools Settings[/]", expand=False))

        menu_items = [
            f"1. Toggle UserInput Tool (current: {'On' if enable_user_input else 'Off'})",
            f"2. Toggle Think Tool (current: {'On' if enable_think else 'Off'})",
            None,
            "3. Back (Save Changes)",
            "4. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 3:
            config_to_edit["tools_settings"] = original_settings
            if not config_to_edit["tools_settings"]:
                config_to_edit.pop("tools_settings", None)
            break
        elif selected_index == 0:
            settings["enable_user_input"] = not enable_user_input
            continue
        elif selected_index == 1:
            settings["enable_think"] = not enable_think
            continue
        elif selected_index == 2:
            break

def _prompt_for_custom_instructions(config_to_edit: dict):
    """Interactively prompts for global custom instructions."""
    console = Console()
    history = InMemoryHistory()
    
    current_instructions = config_to_edit.get("custom_instructions", "")

    while True:
        console.clear()
        console.print(Panel(current_instructions or "[dim]No custom instructions set.[/dim]", title="[bold green]Current Custom Instructions[/]", expand=False, width=100))
        
        console.print("\nEnter new instructions below. Press [bold]ESC[/] followed by [bold]Enter[/] to save and exit.")
        console.print("To clear instructions, save an empty input. To cancel, press [bold]Ctrl+C[/].")
        
        try:
            new_instructions = prompt(
                "> ",
                default=current_instructions,
                multiline=True,
                history=history,
                prompt_continuation="  "
            )
            
            if new_instructions != current_instructions:
                config_to_edit["custom_instructions"] = new_instructions
                console.print("\n[bold green]✔ Custom instructions updated.[/bold green]")
            else:
                console.print("\n[yellow]No changes made.[/yellow]")
            
            console.input("Press Enter to return to settings...")
            break
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Cancelled.[/yellow]")
            console.input("Press Enter to return to settings...")
            break

def _prompt_for_safety_settings(config_to_edit: dict):
    """Interactively prompts for safety and cost settings."""
    console = Console()
    
    settings = config_to_edit.setdefault("safety_settings", {})
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        cost_threshold = settings.get("cost_threshold", 0.0)
        
        cost_display = f"${cost_threshold:.4f}" if cost_threshold > 0 else "Disabled"

        config_view_content = (
            f"[bold cyan]Cost Confirmation Threshold:[/bold cyan] {cost_display}\n"
            "[dim]If the estimated cost of a sub-agent task exceeds this, you will be asked for confirmation.[/dim]"
        )
        console.print(Panel(config_view_content, title="[bold green]Safety & Cost Settings[/]", expand=False))

        menu_items = [
            "1. Edit Cost Threshold",
            None,
            "2. Back (Save Changes)",
            "3. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 2:
            config_to_edit["safety_settings"] = original_settings
            if not config_to_edit["safety_settings"]:
                config_to_edit.pop("safety_settings", None)
            break
        elif selected_index == 0:
            new_threshold_str = console.input("Enter new cost threshold (e.g., 0.01 for 1 cent, 0 to disable): ").strip()
            try:
                new_threshold = float(new_threshold_str)
                if new_threshold < 0: raise ValueError
                settings["cost_threshold"] = new_threshold
            except (ValueError, TypeError):
                console.print("\n[bold red]Invalid input. Please enter a non-negative number.[/bold red]")
                console.input("Press Enter to continue...")
            continue
        elif selected_index == 1:
            break

def _prompt_for_misc_settings(config_to_edit: dict):
    """Interactively prompts for Miscellaneous settings."""
    console = Console()
    
    settings = config_to_edit.setdefault("misc_settings", {})
    original_settings = json.loads(json.dumps(settings))

    while True:
        console.clear()
        show_reasoning = settings.get("show_reasoning", False)

        config_view_content = (
            f"[bold cyan]Show LLM Reasoning:[/bold cyan] {'On' if show_reasoning else 'Off'}"
        )
        console.print(Panel(config_view_content, title="[bold green]Miscellaneous Settings[/]", expand=False))

        menu_items = [
            f"1. Toggle Show Reasoning (current: {'On' if show_reasoning else 'Off'})",
            None,
            "2. Back (Save Changes)",
            "3. Back (Discard Changes)",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 2:
            config_to_edit["misc_settings"] = original_settings
            if not config_to_edit["misc_settings"]:
                config_to_edit.pop("misc_settings", None)
            break
        elif selected_index == 0:
            settings["show_reasoning"] = not show_reasoning
            continue
        elif selected_index == 1:
            break

def _prompt_for_mcp_settings(config_to_edit: dict):
    """Interactively prompts for MCP server configuration."""
    console = Console()
    
    while True:
        console.clear()
        
        # We load fresh each time to reflect any changes from import
        settings = mcp.load_mcp_servers()
        server_list_content = []
        if settings:
            for name, config in sorted(settings.items()):
                server_list_content.append(f"[cyan]{name}[/]: {config.get('url', 'N/A')}")
        else:
            server_list_content.append("[dim]No MCP servers configured.[/dim]")

        console.print(Panel("\n".join(server_list_content), title="[bold green]Configured MCP Servers (All Scopes)[/]", expand=False))
        console.print("[dim]This is a read-only list. Use the CLI to add/remove servers.[/dim]")

        menu_items = [
            "1. Import from Claude Code config",
            None,
            "2. Back to Other Settings",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 1:
            break
        
        if selected_index == 0:
            mcp.copy_claude_code_mcp_config()
            # No need for console input here, it's handled in the function now.
            continue

def _prompt_for_other_settings(config_to_edit: dict):
    """Shows a submenu for various settings."""
    console = Console()
    while True:
        console.clear()
        console.print(Panel("Select a category to configure.", title="[bold green]Other Settings[/]", expand=False))

        menu_items = [
            "1. Custom Instructions",
            "2. RAG Settings",
            "3. Memory Settings",
            "4. Tools Settings",
            "5. Safety & Cost Settings",
            "6. Miscellaneous Settings",
            "7. MCP Server Settings",
            None,
            "8. Back to Main Menu",
        ]

        terminal_menu = TerminalMenu(menu_items, title="Select an option", menu_cursor_style=("fg_green", "bold"), menu_highlight_style=("bg_green", "fg_black"))
        selected_index = terminal_menu.show()

        if selected_index is None or selected_index == 8:
            break
        elif selected_index == 0:
            _prompt_for_custom_instructions(config_to_edit)
        elif selected_index == 1:
            _prompt_for_rag_settings(config_to_edit)
        elif selected_index == 2:
            _prompt_for_memory_settings(config_to_edit)
        elif selected_index == 3:
            _prompt_for_tools_settings(config_to_edit)
        elif selected_index == 4:
            _prompt_for_safety_settings(config_to_edit)
        elif selected_index == 5:
            _prompt_for_misc_settings(config_to_edit)
        elif selected_index == 6:
            _prompt_for_mcp_settings(config_to_edit)


def prompt_for_config() -> dict:
    """Interactively prompts the user to select a mode and then configure it."""
    console = Console()
    original_config = load_config()
    config_to_edit = json.loads(json.dumps(original_config)) # Deep copy

    provider_models = _get_provider_models()
    all_providers = sorted(list(provider_models.keys()))
    all_modes = ["global", "code", "ask", "architect", "agent-maker", "memory"]
    config_to_edit.setdefault("modes", {})

    while True:
        console.clear()
        console.print(Panel("Select a feature to configure, or save/exit.", title="[bold green]Configuration Menu[/]", expand=False))

        modes_cfg = config_to_edit.get("modes", {})
        global_cfg = modes_cfg.get("global", {})
        
        # --- PRESET MENU ---
        presets = config_to_edit.get("presets", {})
        load_preset_item_text = f"Load Preset... ({len(presets)} saved)"
        save_preset_item_text = "Save Current Config as Preset..."
        delete_preset_item_text = "Delete a Preset..."
        
        menu_items = [
            load_preset_item_text,
            save_preset_item_text,
        ]
        if presets:
            menu_items.append(delete_preset_item_text)
        menu_items.append(None) # Separator

        # --- MODE MENU ---
        mode_menu_items = []
        for mode_name in all_modes:
            mode_config = modes_cfg.get(mode_name, {})
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
            mode_menu_items.append(f"Mode: {mode_name.capitalize():<15} ({display_model})")
        
        menu_items.extend(mode_menu_items)
        menu_items.append(None)
        
        weak_model_cfg = config_to_edit.get("weak_model", {})
        weak_model_display = f"{weak_model_cfg.get('provider', '')}/{weak_model_cfg.get('model', '')}" if weak_model_cfg.get('provider') else "Not Configured"
        weak_model_item_text = f"Feature: Weak Model        ({weak_model_display})"
        menu_items.append(weak_model_item_text)

        emb_cfg = config_to_edit.get("embedding", {})
        emb_display = f"{emb_cfg.get('provider', '')}/{emb_cfg.get('model', '')}" if emb_cfg.get('provider') else "Not Configured"
        embedding_item_text = f"Feature: Embedding Models ({emb_display})"
        menu_items.append(embedding_item_text)

        menu_items.append(None)
        other_settings_item_text = "Other Settings..."
        menu_items.append(other_settings_item_text)

        save_item_text = "Save and Exit"
        exit_item_text = "Exit without Saving"
        menu_items.extend([None, save_item_text, exit_item_text])
        
        terminal_menu = TerminalMenu(
            menu_items,
            title="Use UP/DOWN keys to navigate, ENTER to select.",
            menu_cursor="> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("bg_green", "fg_black"),
        )
        selected_index = terminal_menu.show()
        
        selected_item_text = menu_items[selected_index] if selected_index is not None else None

        if selected_item_text is None or selected_item_text == exit_item_text:
            console.print("\n[yellow]Configuration changes discarded.[/yellow]")
            return original_config
        
        if selected_item_text == save_item_text:
            modes_to_del = [m for m, c in config_to_edit.get("modes", {}).items() if not c]
            for m in modes_to_del:
                if m in config_to_edit["modes"]:
                    del config_to_edit["modes"][m]

            save_config(config_to_edit)
            console.print("\n[bold green]✔ Configuration saved successfully.[/bold green]")
            return config_to_edit

        if selected_item_text == load_preset_item_text:
            _prompt_for_load_preset(config_to_edit)
            continue
        elif selected_item_text == save_preset_item_text:
            _prompt_for_save_preset(config_to_edit)
            continue
        elif presets and selected_item_text == delete_preset_item_text:
            _prompt_for_delete_preset(config_to_edit)
            continue
        
        if selected_item_text in mode_menu_items:
            # Find the corresponding mode name
            mode_idx = mode_menu_items.index(selected_item_text)
            selected_mode = all_modes[mode_idx]
            _prompt_for_one_mode(config_to_edit, selected_mode, provider_models, all_providers)
        elif selected_item_text == weak_model_item_text:
            _prompt_for_weak_model_config(config_to_edit, provider_models, all_providers)
        elif selected_item_text == embedding_item_text:
            _prompt_for_embedding_config(config_to_edit, provider_models)
        elif selected_item_text == other_settings_item_text:
            _prompt_for_other_settings(config_to_edit)
