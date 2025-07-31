import pytest
import json
from agentic import config

def test_get_provider_models(mocker, tmp_path):
    """Tests model fetching, filtering, and caching under online/offline scenarios."""
    # --- Setup ---
    mock_online_data = {
        "litellm_spec": "some_spec_data",
        "gpt-4": {"litellm_provider": "openai", "mode": "chat"},
        "text-embedding-ada-002": {"litellm_provider": "openai", "mode": "embedding"},
        "claude-2": {"litellm_provider": "anthropic", "mode": "chat"},
        "replicate/meta/llama-2-70b-chat:abc": {"litellm_provider": "replicate", "mode": "chat"},
    }
    cache_file = tmp_path / "model_cache.json"
    mocker.patch.object(config, "MODELS_CACHE_FILE", cache_file)
    mocker.patch.object(config, "CONFIG_DIR", tmp_path) # For _ensure_config_dir

    # --- 1. Test Online Scenario: Fetches, filters, and writes to cache ---
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_online_data
    mock_response.raise_for_status.return_value = None
    mocker.patch("requests.get", return_value=mock_response)

    provider_models_online = config._get_provider_models()

    assert provider_models_online["openai"] == ["gpt-4"]
    assert provider_models_online["anthropic"] == ["claude-2"]
    assert "text-embedding-ada-002" not in provider_models_online.get("openai", [])
    assert cache_file.exists() and json.loads(cache_file.read_text()) == mock_online_data

    # --- 2. Test Offline Scenario with Cache ---
    mock_get_offline = mocker.patch("requests.get", side_effect=config.requests.RequestException("Network Error"))

    provider_models_offline_cache = config._get_provider_models()

    # Should return the same filtered models from the cache written in the previous step
    assert provider_models_offline_cache == provider_models_online
    mock_get_offline.assert_called_once() # Ensure it tried to fetch first

    # --- 3. Test Offline Scenario without Cache ---
    cache_file.unlink() # Delete the cache
    assert not cache_file.exists()
    
    provider_models_offline_no_cache = config._get_provider_models()
    
    assert provider_models_offline_no_cache == {}

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
