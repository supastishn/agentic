import pytest
from agentic.cli import _should_add_to_history, is_config_valid

@pytest.mark.parametrize("text, expected", [
    ("hello world", True),
    ("  ", False),
    ("", False),
    ("/help", False),
    ("!ls", False),
    ("exit", False),
])
def test_should_add_to_history(text, expected):
    assert _should_add_to_history(text) == expected

def test_is_config_valid():
    # Test valid config with at least one mode configured
    valid_config = {
        "modes": {
            "code": {
                "active_provider": "openai",
                "providers": {"openai": {"model": "gpt-4", "api_key": "sk-123"}}
            },
            "ask": {}
        }
    }
    assert is_config_valid(valid_config) is True

    # Test valid config (mode configured but missing key)
    invalid_key = {
        "modes": {
            "code": {
                "active_provider": "openai",
                "providers": {"openai": {"model": "gpt-4"}}
            }
        }
    }
    assert is_config_valid(invalid_key) is True

    # Test invalid config (mode configured but missing model)
    invalid_model = {
        "modes": {
            "code": {
                "active_provider": "openai",
                "providers": {"openai": {"api_key": "sk-123"}}
            }
        }
    }
    assert is_config_valid(invalid_model) is False
    
    # Test invalid config (no modes configured)
    invalid_no_modes_configured = {
        "modes": {
            "code": {},
            "ask": {"active_provider": "openai"}
        }
    }
    assert is_config_valid(invalid_no_modes_configured) is False

    # Test empty config
    empty_config = {}
    assert is_config_valid(empty_config) is False
    
    # Test config with empty modes dict
    empty_modes = {"modes": {}}
    assert is_config_valid(empty_modes) is False
