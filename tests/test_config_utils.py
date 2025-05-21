import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import shutil # shutil is used by the module, so it's good to be aware for mocking
from pathlib import Path
import logging # For testing logging calls

# Ensure the test can find the module
# This assumes tests/ is at the same level as src/
# Adjust if your project structure is different.
sys_path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if sys_path_to_add not in sys.path:
    sys.path.insert(0, sys_path_to_add)

# Now import the module under test
from src.dolphin_logger import config_utils

# --- Fixtures ---

@pytest.fixture
def mock_home_dir(tmp_path):
    """Fixture to create a temporary mock home directory."""
    home_d = tmp_path / "mock_home"
    home_d.mkdir()
    return home_d

@pytest.fixture
def mock_config_dir_path(mock_home_dir):
    """Path to the mock .dolphin-logger directory."""
    return mock_home_dir / ".dolphin-logger"

@pytest.fixture
def mock_logs_dir_path(mock_config_dir_path):
    """Path to the mock logs directory."""
    return mock_config_dir_path / "logs"

@pytest.fixture
def mock_user_config_file_path(mock_config_dir_path):
    """Path to the mock user config.json file."""
    return mock_config_dir_path / "config.json"

# --- Tests for get_config_dir ---
def test_get_config_dir(mocker, mock_home_dir, mock_config_dir_path):
    mocker.patch.object(Path, 'home', return_value=mock_home_dir)
    
    # We also need to mock mkdir for Path objects if it's called by the function
    mock_mkdir = mocker.patch.object(Path, 'mkdir')

    actual_path = config_utils.get_config_dir()
    
    assert actual_path == mock_config_dir_path
    # Check that Path.home().mkdir was called correctly for .dolphin-logger
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# --- Tests for get_config_path ---
def test_get_config_path(mocker, mock_config_dir_path, mock_user_config_file_path):
    # Mock get_config_dir to return our controlled path
    mocker.patch.object(config_utils, 'get_config_dir', return_value=mock_config_dir_path)
    
    actual_path = config_utils.get_config_path()
    
    assert actual_path == mock_user_config_file_path

# --- Tests for get_logs_dir ---
def test_get_logs_dir(mocker, mock_config_dir_path, mock_logs_dir_path):
    mocker.patch.object(config_utils, 'get_config_dir', return_value=mock_config_dir_path)
    
    # Mock mkdir for the logs directory itself
    mock_logs_mkdir = mocker.patch.object(Path, 'mkdir')
    
    # Temporarily assign to a variable that matches the one used in get_logs_dir
    # if Path object's mkdir is called on the logs_dir path itself.
    # Here, we assume get_config_dir returns a Path obj, and then / "logs" is also a Path obj
    # and then .mkdir() is called on *that*.
    
    # We need to ensure that when `logs_dir.mkdir` is called, our mock is used.
    # If `get_config_dir()` returns a Path object, then `logs_dir = config_dir / "logs"`
    # will also be a Path object. Its `mkdir` method needs to be mocked.
    # The easiest way is to patch Path.mkdir generally as it's specific to this path.
    
    # Reset general Path.mkdir mock if it was set by get_config_dir test
    if 'Path.mkdir' in mocker. reciente_mocks: # pytest-mock specific
        mocker.stopall() # Stop all general mocks to be safe
        mocker.patch.object(config_utils, 'get_config_dir', return_value=mock_config_dir_path) # Re-patch this
        mock_logs_mkdir = mocker.patch.object(Path, 'mkdir') # Re-patch this specifically for logs_dir

    actual_path = config_utils.get_logs_dir()
    
    assert actual_path == mock_logs_dir_path
    # The mkdir call for logs_dir itself
    mock_logs_mkdir.assert_called_with(parents=True, exist_ok=True)


# --- Tests for load_config ---

# Scenario 1: Default Config Creation
@patch('src.dolphin_logger.config_utils.shutil.copy')
@patch('src.dolphin_logger.config_utils.Path.exists')
def test_load_config_creates_default_when_not_exists(
    mock_path_exists, mock_shutil_copy, mocker, 
    mock_user_config_file_path, mock_config_dir_path # Use fixtures
):
    mock_path_exists.return_value = False # Simulate user config.json does not exist
    
    # Mock get_config_path to return our controlled user config file path
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    
    # The actual default config.json path within the package
    # Path(__file__).parent refers to tests directory.
    # So, ../src/dolphin_logger/config.json
    package_default_config_path = Path(__file__).resolve().parent.parent / "src" / "dolphin_logger" / "config.json"

    default_config_content = {"models": [{"provider": "default", "model": "default_model", "apiKey": "default_key"}]}
    
    # Mock 'open' to simulate reading the default config *after* it's supposedly copied
    # So, when open is called for user_config_file_path, it should return this content.
    m = mock_open(read_data=json.dumps(default_config_content))
    mocker.patch('builtins.open', m)

    config = config_utils.load_config()

    mock_shutil_copy.assert_called_once_with(package_default_config_path, mock_user_config_file_path)
    m.assert_called_once_with(mock_user_config_file_path, "r")
    assert config == default_config_content

# Scenario 2: Loading Existing Config
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True) # Config exists
@patch('src.dolphin_logger.config_utils.shutil.copy') # Should not be called
def test_load_config_loads_existing(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    existing_config_content = {"models": [{"provider": "openai", "model": "gpt-test", "apiKey": "existing_key"}]}
    
    m = mock_open(read_data=json.dumps(existing_config_content))
    mocker.patch('builtins.open', m)

    config = config_utils.load_config()

    mock_shutil_copy.assert_not_called()
    m.assert_called_once_with(mock_user_config_file_path, "r")
    assert config == existing_config_content

# Scenario 3: API Key Resolution from Environment Variables
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_api_key_env_var_resolution(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    
    TEST_API_KEY_NAME = "MY_TEST_API_KEY"
    TEST_API_KEY_VALUE = "resolved_secret_key"

    config_with_env_ref = {
        "models": [
            {"provider": "openai", "model": "gpt-env", "apiKey": f"ENV:{TEST_API_KEY_NAME}"},
            {"provider": "anthropic", "model": "claude-direct", "apiKey": "actual_claude_key"}
        ]
    }
    expected_resolved_config = {
        "models": [
            {"provider": "openai", "model": "gpt-env", "apiKey": TEST_API_KEY_VALUE},
            {"provider": "anthropic", "model": "claude-direct", "apiKey": "actual_claude_key"}
        ]
    }

    m = mock_open(read_data=json.dumps(config_with_env_ref))
    mocker.patch('builtins.open', m)
    
    # Mock os.environ.get
    mock_environ_get = mocker.patch.dict(os.environ, {TEST_API_KEY_NAME: TEST_API_KEY_VALUE})

    config = config_utils.load_config()

    assert config == expected_resolved_config
    mock_shutil_copy.assert_not_called()

# Scenario 4: API Key Resolution - Environment Variable Not Found
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
@patch('builtins.print') # The code uses print for warnings
def test_load_config_api_key_env_var_not_found(
    mock_print, mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    
    MISSING_KEY_NAME = "MY_MISSING_API_KEY"
    config_with_missing_env = {
        "models": [
            {"provider": "openai", "model": "gpt-missing-env", "apiKey": f"ENV:{MISSING_KEY_NAME}"}
        ]
    }
    # According to current implementation, apiKey becomes None if env var is missing
    expected_config_missing_resolved = {
        "models": [
            {"provider": "openai", "model": "gpt-missing-env", "apiKey": None}
        ]
    }

    m = mock_open(read_data=json.dumps(config_with_missing_env))
    mocker.patch('builtins.open', m)
    
    # Ensure the env var is not set
    mocker.patch.dict(os.environ, {}, clear=True)

    config = config_utils.load_config()

    assert config == expected_config_missing_resolved
    # Check that print was called with a warning
    mock_print.assert_any_call(f"Warning: Environment variable '{MISSING_KEY_NAME}' not found for model 'gpt-missing-env'. API key set to None.")
    mock_shutil_copy.assert_not_called()

# Scenario 5: Handling Malformed JSON
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_handles_malformed_json(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    
    malformed_json_content = '{"models": [{"provider": "test", "model": "bad_json"},,,]}' # Invalid JSON
    
    m = mock_open(read_data=malformed_json_content)
    mocker.patch('builtins.open', m)
    
    # Mock json.load to raise JSONDecodeError when called by the SUT's open mock
    mocker.patch('json.load', side_effect=json.JSONDecodeError("Test SytaxError", "doc", 0))

    with pytest.raises(json.JSONDecodeError) as excinfo:
        config_utils.load_config()
            
    # The current code re-raises with a more specific message.
    # We check if the original error type is json.JSONDecodeError.
    # The message check might be too brittle if the SUT changes its error message format.
    # assert "Error decoding JSON from config file" in str(excinfo.value)
    assert excinfo.type == json.JSONDecodeError
    mock_shutil_copy.assert_not_called()

# Scenario 6: Handling Missing `models` Key
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_missing_models_key(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    config_without_models = {"some_other_key": "some_value"} # No 'models' key
    
    m = mock_open(read_data=json.dumps(config_without_models))
    mocker.patch('builtins.open', m)

    config = config_utils.load_config()

    # The current implementation of load_config processes 'models' if it exists.
    # If 'models' key is missing, it should return the dict as is.
    # The environment variable processing part will be skipped.
    assert config == config_without_models
    mock_shutil_copy.assert_not_called()

# Additional test: Empty JSON file
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_empty_json_file(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    empty_json_content = "{}"
    m = mock_open(read_data=empty_json_content)
    mocker.patch('builtins.open', m)
    config = config_utils.load_config()
    assert config == {} # Should load as an empty dict
    mock_shutil_copy.assert_not_called()

# Additional test: 'models' is not a list
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_models_not_a_list(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": "this is not a list"} # 'models' is a string
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = config_utils.load_config()
    # Env var processing should be skipped, config returned as is.
    assert config == {"models": "this is not a list"}
    mock_shutil_copy.assert_not_called()

# Additional test: Model entry in 'models' list is not a dictionary
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_model_entry_not_a_dict(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": ["not_a_dict_entry", {"provider": "openai", "apiKey": "key"}]}
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = config_utils.load_config()
    # Env var processing skips non-dict entries.
    assert config == {"models": ["not_a_dict_entry", {"provider": "openai", "apiKey": "key"}]}
    mock_shutil_copy.assert_not_called()

# Additional test: apiKey in a model is not a string
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
def test_load_config_api_key_not_a_string(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": [{"provider": "openai", "apiKey": 12345}]} # apiKey is a number
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = config_utils.load_config()
    # Env var processing skips non-string apiKeys.
    assert config == {"models": [{"provider": "openai", "apiKey": 12345}]}
    mock_shutil_copy.assert_not_called()

# Test for config where apiKey is "ENV:" but no var name follows
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=True)
@patch('src.dolphin_logger.config_utils.shutil.copy')
@patch('builtins.print') # Mock print to check for warnings
def test_load_config_api_key_env_empty_var_name(
    mock_print, mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": [{"provider": "openai", "model": "test_model", "apiKey": "ENV:"}]}
    expected_config = {"models": [{"provider": "openai", "model": "test_model", "apiKey": None}]}
    
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    mocker.patch.dict(os.environ, {}, clear=True) # Ensure no relevant env vars

    config = config_utils.load_config()
    
    assert config == expected_config
    mock_shutil_copy.assert_not_called()
    
    # Check that the warning about empty env var name was printed
    mock_print.assert_any_call("Warning: Environment variable '' not found for model 'test_model'. API key set to None.")

# Test case for FileNotFoundError if default config is also missing (highly unlikely but good for robustness)
@patch('src.dolphin_logger.config_utils.Path.exists', return_value=False) # User config does not exist
@patch('src.dolphin_logger.config_utils.shutil.copy', side_effect=FileNotFoundError("Mock: Default package config missing"))
def test_load_config_default_config_missing_shutil_error(
    mock_shutil_copy_error, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(config_utils, 'get_config_path', return_value=mock_user_config_file_path)
    
    # Since shutil.copy fails, the subsequent open() for the user_config_file_path will also fail
    # because the file was never created. The SUT catches FileNotFoundError from open().
    mocker.patch('builtins.open', mock_open(side_effect=FileNotFoundError))

    with pytest.raises(FileNotFoundError) as excinfo:
        config_utils.load_config()
    
    # The error should originate from the attempt to open the user_config_file after copy failed.
    assert f"Config file not found at {mock_user_config_file_path}" in str(excinfo.value)
    # Or, depending on how it's caught and re-raised:
    # assert "Mock: Default package config missing" in str(excinfo.value)


# Test that pkg_resources is not used (as it was removed from requirements)
def test_pkg_resources_not_used(mocker):
    # If pkg_resources was imported and used, this would try to mock it.
    # The goal is to ensure it's NOT used.
    # A bit of a meta-test: if the import itself fails in config_utils, this test might not catch it
    # directly, but rather the module import at the top of this test file would fail.
    # This test mainly ensures no dynamic `pkg_resources.resource_filename` calls are made.
    mock_pkg_resources = MagicMock()
    mocker.patch.dict(sys.modules, {'pkg_resources': mock_pkg_resources})
    
    # Call a function that might have used it (load_config is the candidate)
    # We need to mock other things for load_config to run without error
    with patch('src.dolphin_logger.config_utils.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='{}')), \
         patch.object(config_utils, 'get_config_path', return_value=Path("dummy/path/config.json")):
        config_utils.load_config()
    
    mock_pkg_resources.resource_filename.assert_not_called()
