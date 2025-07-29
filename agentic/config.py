import os
import json
from pathlib import Path
import sys
from cryptography.fernet import Fernet
from rich.console import Console
from rich.panel import Panel

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
    # Use a copy to allow for discarding changes
    config_to_edit = original_config.copy()

    while True:
        console.clear()

        # Get current values for display
        model = config_to_edit.get("model", "gpt-4o")
        api_key = config_to_edit.get("api_key")
        api_key_display = f"****{api_key[-4:]}" if api_key else "Not set"

        config_view_content = (
            f"[bold cyan]Model:[/bold cyan] {model}\n"
            f"[bold cyan]API Key:[/bold cyan] {api_key_display}"
        )
        console.print(Panel(config_view_content, title="[bold green]Current Configuration[/]", expand=False))

        menu_content = (
            "1. Edit Model\n"
            "2. Edit API Key\n"
            "\n"
            "3. [bold green]Save and Exit[/bold green]\n"
            "4. [yellow]Exit without Saving[/yellow]"
        )
        console.print(Panel(menu_content, title="[bold yellow]Configuration Menu[/]", expand=False))

        choice = console.input("[bold]Enter your choice (1-4):[/bold] ").strip()

        if choice == '1':
            new_model = console.input(f"Enter new model ([default]{model}[/default]): ").strip()
            if new_model:
                config_to_edit["model"] = new_model
        elif choice == '2':
            new_api_key = console.input(f"Enter new API Key ([default]{api_key_display}[/default]): ").strip()
            if new_api_key:
                config_to_edit["api_key"] = new_api_key
        elif choice == '3':  # Save and Exit
            if not config_to_edit.get("api_key"):
                console.print("[bold red]API Key is required. Configuration not saved.[/bold red]")
                console.input("Press Enter to continue...")
                continue
            save_config(config_to_edit)
            console.print("\n[bold green]✔ Configuration saved successfully.[/bold green]")
            return config_to_edit
        elif choice == '4':  # Exit without Saving
            console.print("\n[yellow]Configuration changes discarded.[/yellow]")
            return original_config
        else:
            console.print(f"[bold red]Invalid choice: '{choice}'. Please try again.[/bold red]")
            console.input("Press Enter to continue...")
