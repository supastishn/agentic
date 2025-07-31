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
    # Test valid new config
    valid_new = {
        "active_provider": "openai",
        "providers": {
            "openai": {"model": "gpt-4", "api_key": "sk-123"}
        }
    }
    assert is_config_valid(valid_new) is True

    # Test invalid new config (missing model)
    invalid_new_model = {
        "active_provider": "openai",
        "providers": {"openai": {"api_key": "sk-123"}}
    }
    assert is_config_valid(invalid_new_model) is False

    # Test invalid new config (missing key)
    invalid_new_key = {
        "active_provider": "openai",
        "providers": {"openai": {"model": "gpt-4"}}
    }
    assert is_config_valid(invalid_new_key) is False

    # Test valid old config
    valid_old = {"api_key": "sk-123"}
    assert is_config_valid(valid_old) is True
    
    # Test invalid old config
    invalid_old = {"model": "gpt-4"}
    assert is_config_valid(invalid_old) is False
    
    # Test empty config
    empty_config = {}
    assert is_config_valid(empty_config) is False
