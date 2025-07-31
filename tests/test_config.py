import pytest
import json
from agentic import config

def test_get_provider_models(mocker):
    """Tests if models are correctly parsed and grouped by provider from fetched JSON."""
    mock_json_data = {
        "gpt-4": {"litellm_provider": "openai"},
        "gpt-3.5-turbo": {"litellm_provider": "openai"},
        "claude-2": {"litellm_provider": "anthropic"},
        "replicate/meta/llama-2-70b-chat:abc": {"litellm_provider": "replicate"},
        "some-model-without-provider": {},
        "openai/test/with/slashes": {"litellm_provider": "openai"},
    }

    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_json_data
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    provider_models = config._get_provider_models()

    assert "openai" in provider_models
    assert "anthropic" in provider_models
    assert "replicate" in provider_models
    assert "some-model-without-provider" not in provider_models

    assert provider_models["openai"] == [
        "gpt-3.5-turbo",
        "gpt-4",
        "test/with/slashes",
    ]
    assert provider_models["anthropic"] == ["claude-2"]
    assert provider_models["replicate"] == ["meta/llama-2-70b-chat:abc"]

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
