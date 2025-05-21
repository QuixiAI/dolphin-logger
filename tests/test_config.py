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
from src.dolphin_logger import config as dolphin_config_module

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

    actual_path = dolphin_config_module.get_config_dir()
    
    assert actual_path == mock_config_dir_path
    # Check that Path.home().mkdir was called correctly for .dolphin-logger
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# --- Tests for get_config_path ---
def test_get_config_path(mocker, mock_config_dir_path, mock_user_config_file_path):
    # Mock get_config_dir to return our controlled path
    mocker.patch.object(dolphin_config_module, 'get_config_dir', return_value=mock_config_dir_path)
    
    actual_path = dolphin_config_module.get_config_path()
    
    assert actual_path == mock_user_config_file_path

# --- Tests for get_logs_dir ---
def test_get_logs_dir(mocker, mock_config_dir_path, mock_logs_dir_path):
    mocker.patch.object(dolphin_config_module, 'get_config_dir', return_value=mock_config_dir_path)
    
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
    if 'Path.mkdir' in mocker. 최근_mocks: # Use the correct attribute for recent mocks if necessary
        mocker.stopall() # Stop all general mocks to be safe
        mocker.patch.object(dolphin_config_module, 'get_config_dir', return_value=mock_config_dir_path) # Re-patch this
        mock_logs_mkdir = mocker.patch.object(Path, 'mkdir') # Re-patch this specifically for logs_dir
    else: # If no recent mocks or attribute name is different, just patch
        mock_logs_mkdir = mocker.patch.object(Path, 'mkdir')


    actual_path = dolphin_config_module.get_logs_dir()
    
    assert actual_path == mock_logs_dir_path
    # The mkdir call for logs_dir itself
    mock_logs_mkdir.assert_called_with(parents=True, exist_ok=True)


# --- Tests for load_config ---

# Scenario 1: Default Config Creation
@patch('src.dolphin_logger.config.shutil.copy') # Path to shutil in config.py
@patch('src.dolphin_logger.config.Path.exists')   # Path to Path in config.py
def test_load_config_creates_default_when_not_exists(
    mock_path_exists, mock_shutil_copy, mocker, 
    mock_user_config_file_path, mock_config_dir_path
):
    mock_path_exists.return_value = False 
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    # Path to the default config.json within the package, relative to config.py
    # config.py is in src/dolphin_logger, so Path(__file__).parent is src/dolphin_logger
    # Path(__file__).parent refers to tests directory.
    # So, ../src/dolphin_logger/config.json
    # This needs to align with how config.py finds its *sibling* config.json (the default one)
    # In config.py: package_config_path = Path(__file__).parent / "config.json"
    # So, when config.py is src/dolphin_logger/config.py, Path(__file__).parent is src/dolphin_logger
    # and it looks for src/dolphin_logger/config.json
    # This test needs to mock the existence and copy of *that* file.
    # The `package_default_config_path` variable here is for the shutil.copy assertion.
    # The actual `Path(__file__).parent / "config.json"` will be executed inside `load_config`.
    
    # We need to provide a mock for `Path(__file__).parent / "config.json"` if it's different
    # from the `package_default_config_path` used for assertion.
    # For `shutil.copy` assertion, the source path must be correct.
    # `Path(__file__).parent` inside `config.py` refers to `src/dolphin_logger`.
    # So, `package_config_path` in `load_config` will be `src/dolphin_logger/config.json`.
    # This should be the source for `shutil.copy`.
    expected_package_default_path = Path(dolphin_config_module.__file__).parent / "config.json"


    default_config_content = {"models": [{"provider": "default", "model": "default_model", "apiKey": "default_key"}]}
    m = mock_open(read_data=json.dumps(default_config_content))
    mocker.patch('builtins.open', m)

    config = dolphin_config_module.load_config()

    mock_shutil_copy.assert_called_once_with(expected_package_default_path, mock_user_config_file_path)
    m.assert_called_once_with(mock_user_config_file_path, "r")
    assert config == default_config_content

# Scenario 2: Loading Existing Config
@patch('src.dolphin_logger.config.Path.exists', return_value=True) 
@patch('src.dolphin_logger.config.shutil.copy') 
def test_load_config_loads_existing(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    existing_config_content = {"models": [{"provider": "openai", "model": "gpt-test", "apiKey": "existing_key"}]}
    
    m = mock_open(read_data=json.dumps(existing_config_content))
    mocker.patch('builtins.open', m)

    config = dolphin_config_module.load_config()

    mock_shutil_copy.assert_not_called()
    m.assert_called_once_with(mock_user_config_file_path, "r")
    assert config == existing_config_content

# Scenario 3: API Key Resolution from Environment Variables
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_api_key_env_var_resolution(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    TEST_API_KEY_NAME = "MY_CONFIG_TEST_API_KEY" # Renamed to avoid potential collision
    TEST_API_KEY_VALUE = "resolved_config_secret_key"

    config_with_env_ref = {
        "models": [
            {"provider": "openai", "model": "gpt-env-config", "apiKey": f"ENV:{TEST_API_KEY_NAME}"},
            {"provider": "anthropic", "model": "claude-direct-config", "apiKey": "actual_claude_key_config"}
        ]
    }
    expected_resolved_config = {
        "models": [
            {"provider": "openai", "model": "gpt-env-config", "apiKey": TEST_API_KEY_VALUE},
            {"provider": "anthropic", "model": "claude-direct-config", "apiKey": "actual_claude_key_config"}
        ]
    }

    m = mock_open(read_data=json.dumps(config_with_env_ref))
    mocker.patch('builtins.open', m)
    mocker.patch.dict(os.environ, {TEST_API_KEY_NAME: TEST_API_KEY_VALUE})

    config = dolphin_config_module.load_config()

    assert config == expected_resolved_config
    mock_shutil_copy.assert_not_called()

# Scenario 4: API Key Resolution - Environment Variable Not Found
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
@patch('builtins.print') 
def test_load_config_api_key_env_var_not_found(
    mock_print, mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    MISSING_KEY_NAME = "MY_CONFIG_MISSING_API_KEY"
    config_with_missing_env = {
        "models": [
            {"provider": "openai", "model": "gpt-missing-env-config", "apiKey": f"ENV:{MISSING_KEY_NAME}"}
        ]
    }
    expected_config_missing_resolved = {
        "models": [
            {"provider": "openai", "model": "gpt-missing-env-config", "apiKey": None}
        ]
    }

    m = mock_open(read_data=json.dumps(config_with_missing_env))
    mocker.patch('builtins.open', m)
    mocker.patch.dict(os.environ, {}, clear=True)

    config = dolphin_config_module.load_config()

    assert config == expected_config_missing_resolved
    mock_print.assert_any_call(f"Warning: Environment variable '{MISSING_KEY_NAME}' not found for model 'gpt-missing-env-config'. API key set to None.")
    mock_shutil_copy.assert_not_called()

# Scenario 5: Handling Malformed JSON
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_handles_malformed_json(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    malformed_json_content = '{"models": [{"provider": "test", "model": "bad_json_config"},,,]}'
    
    m = mock_open(read_data=malformed_json_content)
    mocker.patch('builtins.open', m)
    # json.load is called inside the SUT, so if the read_data is malformed, it will raise naturally.
    # However, the SUT catches this and re-raises. To test the re-raise, we can mock json.load.
    mocker.patch('json.load', side_effect=json.JSONDecodeError("Test SyntaxError for config", "doc", 0))

    with pytest.raises(json.JSONDecodeError) as excinfo:
        dolphin_config_module.load_config()
            
    # The error message from SUT is "Error decoding JSON from config file at {config_path}"
    # This test ensures that the specific error type is raised.
    assert "Error decoding JSON from config file at" in str(excinfo.value) 
    # assert excinfo.type == json.JSONDecodeError # This is also true
    mock_shutil_copy.assert_not_called()

# Scenario 6: Handling Missing `models` Key
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_missing_models_key(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    config_without_models = {"some_other_key": "some_value_config"}
    
    m = mock_open(read_data=json.dumps(config_without_models))
    mocker.patch('builtins.open', m)

    config = dolphin_config_module.load_config()

    assert config == config_without_models
    mock_shutil_copy.assert_not_called()

# Additional test: Empty JSON file
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_empty_json_file(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    empty_json_content = "{}"
    m = mock_open(read_data=empty_json_content)
    mocker.patch('builtins.open', m)
    config = dolphin_config_module.load_config()
    assert config == {}
    mock_shutil_copy.assert_not_called()

# Additional test: 'models' is not a list
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_models_not_a_list(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": "this is not a list config"}
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = dolphin_config_module.load_config()
    assert config == {"models": "this is not a list config"}
    mock_shutil_copy.assert_not_called()

# Additional test: Model entry in 'models' list is not a dictionary
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_model_entry_not_a_dict(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": ["not_a_dict_entry_config", {"provider": "openai", "apiKey": "key_config"}]}
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = dolphin_config_module.load_config()
    assert config == {"models": ["not_a_dict_entry_config", {"provider": "openai", "apiKey": "key_config"}]}
    mock_shutil_copy.assert_not_called()

# Additional test: apiKey in a model is not a string
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
def test_load_config_api_key_not_a_string(
    mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": [{"provider": "openai", "apiKey": 123456}]} # apiKey is a number
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    config = dolphin_config_module.load_config()
    assert config == {"models": [{"provider": "openai", "apiKey": 123456}]}
    mock_shutil_copy.assert_not_called()

# Test for config where apiKey is "ENV:" but no var name follows
@patch('src.dolphin_logger.config.Path.exists', return_value=True)
@patch('src.dolphin_logger.config.shutil.copy')
@patch('builtins.print') 
def test_load_config_api_key_env_empty_var_name(
    mock_print, mock_shutil_copy, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    config_content = {"models": [{"provider": "openai", "model": "test_model_config_env_empty", "apiKey": "ENV:"}]}
    expected_config = {"models": [{"provider": "openai", "model": "test_model_config_env_empty", "apiKey": None}]}
    
    m = mock_open(read_data=json.dumps(config_content))
    mocker.patch('builtins.open', m)
    mocker.patch.dict(os.environ, {}, clear=True)

    config = dolphin_config_module.load_config()
    
    assert config == expected_config
    mock_shutil_copy.assert_not_called()
    mock_print.assert_any_call("Warning: Environment variable '' not found for model 'test_model_config_env_empty'. API key set to None.")

# Test case for FileNotFoundError if default config is also missing
@patch('src.dolphin_logger.config.Path.exists', return_value=False) 
@patch('src.dolphin_logger.config.shutil.copy', side_effect=FileNotFoundError("Mock: Default package config.json missing"))
def test_load_config_default_config_missing_shutil_error(
    mock_shutil_copy_error, mock_path_exists, mocker, mock_user_config_file_path
):
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    # SUT's open will fail as shutil.copy failed.
    mocker.patch('builtins.open', mock_open(side_effect=FileNotFoundError))

    with pytest.raises(FileNotFoundError) as excinfo:
        dolphin_config_module.load_config()
    
    assert f"Config file not found at {mock_user_config_file_path}" in str(excinfo.value)


# Test that importlib.resources is used instead of pkg_resources
@patch('src.dolphin_logger.config.importlib.resources.files')
def test_uses_importlib_resources(mock_importlib_files, mocker, mock_user_config_file_path):
    # This test ensures that importlib.resources.files is called,
    # implying pkg_resources is not used for finding the default config.
    
    # Simulate user config not existing to trigger default config copy logic
    mocker.patch.object(Path, 'exists', return_value=False) 
    mocker.patch.object(dolphin_config_module, 'get_config_path', return_value=mock_user_config_file_path)
    
    # Mock the traversable object returned by importlib.resources.files(...).joinpath(...)
    mock_traversable = MagicMock()
    mock_traversable.is_file.return_value = True # Simulate default config.json.example exists
    mock_importlib_files.return_value.joinpath.return_value = mock_traversable
    
    # Mock shutil.copy and open as they are also part of this flow
    mocker.patch('src.dolphin_logger.config.shutil.copy')
    mocker.patch('builtins.open', mock_open(read_data='{"models":[]}'))

    dolphin_config_module.load_config()
    
    mock_importlib_files.assert_called_once_with('dolphin_logger')
    mock_importlib_files.return_value.joinpath.assert_called_once_with('config.json')
