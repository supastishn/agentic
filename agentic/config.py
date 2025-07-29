import os
import json
from pathlib import Path
import sys
from cryptography.fernet import Fernet

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
    """Interactively prompts the user for configuration settings and saves them."""
    current_config = load_config()
    print("\n--- Configure Agent ---")
    print("Press Enter to keep the current value.")

    # Model
    current_model = current_config.get("model", "gpt-4o")
    model = input(f"Model [{current_model}]: ").strip() or current_model

    # API Key
    api_key_val = current_config.get("api_key")
    api_key_display = f"****{api_key_val[-4:]}" if api_key_val else "Not set"
    api_key = input(f"API Key [{api_key_display}]: ").strip() or api_key_val

    if not api_key:
        print("API Key is required. Configuration not saved.", file=sys.stderr)
        return current_config

    updated_config = {"model": model, "api_key": api_key}
    save_config(updated_config)
    print("Configuration saved successfully.")
    return updated_config
