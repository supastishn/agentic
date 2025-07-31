import pytest
import json
from agentic import config

def test_get_provider_models(mocker):
    """Tests if models are correctly parsed and grouped by provider."""
    mock_model_list = [
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "anthropic/claude-2",
        "google/gemini-pro",
        "mistral/mistral-7b-instruct",
        "nomodelprovider", # should be ignored
        "openai/test/with/slashes"
    ]
    mocker.patch("litellm.model_list", mock_model_list)
    
    provider_models = config._get_provider_models()
    
    assert "openai" in provider_models
    assert "anthropic" in provider_models
    assert "nomodelprovider" not in provider_models
    assert provider_models["openai"] == ["gpt-3.5-turbo", "gpt-4", "test/with/slashes"]
    assert provider_models["anthropic"] == ["claude-2"]

def test_config_encryption_cycle(monkeypatch, tmp_path):
    """Tests saving and loading an encrypted config."""
    # Patch config paths to use a temporary directory
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.encrypted")
    monkeypatch.setattr(config, "KEY_FILE", tmp_path / "config.key")

    # 1. Test saving a new config
    sample_config = {"active_provider": "openai", "providers": {"openai": {"api_key": "sk-1234"}}}
    config.save_config(sample_config)

    assert (tmp_path / "config.key").exists()
    assert (tmp_path / "config.encrypted").exists()
    
    # Ensure file is not plaintext
    encrypted_data = (tmp_path / "config.encrypted").read_text()
    assert "sk-1234" not in encrypted_data

    # 2. Test loading the config
    loaded_config = config.load_config()
    assert loaded_config == sample_config
