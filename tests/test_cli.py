import pytest
from unittest.mock import patch, MagicMock, call
import os
import json
from pathlib import Path
import sys
import argparse # For checking Namespace objects

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the CLI main function and specific command handlers if needed for direct testing
from dolphin_logger.cli import main_cli, _handle_init_command, _handle_config_command
# Import other modules that cli.py interacts with to mock them
from dolphin_logger import config as dolphin_config_module
from dolphin_logger import server as dolphin_server_module
from dolphin_logger import upload as dolphin_upload_module

# --- Fixtures for CLI tests ---
@pytest.fixture
def mock_args():
    """Returns a MagicMock that can be used to simulate parsed args."""
    return MagicMock()

# --- Tests for CLI Dispatch Logic ---
@patch('dolphin_logger.cli.run_server_main')
@patch('dolphin_logger.cli.upload_logs')
@patch('dolphin_logger.cli._handle_init_command')
@patch('dolphin_logger.cli._handle_config_command')
@patch('dolphin_logger.cli.load_config') # Mock load_config called at the start of some commands
@patch('argparse.ArgumentParser.parse_args')
def test_main_cli_dispatch(
    mock_parse_args, mock_load_config, mock_handle_config, 
    mock_handle_init, mock_upload_logs, mock_run_server_main
):
    # Test server command (default)
    mock_parse_args.return_value = argparse.Namespace(command=None) # Simulates no command
    main_cli()
    mock_run_server_main.assert_called_once()
    mock_load_config.assert_called_with() # Called for server command
    mock_run_server_main.reset_mock() # Reset for next call
    mock_load_config.reset_mock()

    # Test server command (explicit)
    mock_parse_args.return_value = argparse.Namespace(command='server')
    main_cli()
    mock_run_server_main.assert_called_once()
    mock_load_config.assert_called_with()
    mock_run_server_main.reset_mock()
    mock_load_config.reset_mock()

    # Test upload command
    mock_parse_args.return_value = argparse.Namespace(command='upload')
    main_cli()
    mock_upload_logs.assert_called_once()
    mock_load_config.assert_called_with() 
    mock_upload_logs.reset_mock()
    mock_load_config.reset_mock()

    # Test init command
    mock_parse_args.return_value = argparse.Namespace(command='init')
    main_cli()
    mock_handle_init.assert_called_once()
    # load_config is NOT called before dispatching to _handle_init_command in current cli.py logic
    mock_load_config.assert_not_called() 
    mock_handle_init.reset_mock()

    # Test config command
    # _handle_config_command takes args, so ensure Namespace has relevant fields
    mock_config_args = argparse.Namespace(command='config', path=False, validate=False) 
    mock_parse_args.return_value = mock_config_args
    main_cli()
    mock_handle_config.assert_called_once_with(mock_config_args)
    mock_load_config.assert_not_called() # Not called before _handle_config_command
    mock_handle_config.reset_mock()


# --- Tests for `init` command handler (_handle_init_command) ---
@patch('dolphin_logger.cli.get_config_dir')
@patch('dolphin_logger.cli.get_config_path')
@patch('dolphin_logger.cli.pkg_resources.files') # Patch importlib.resources.files
@patch('dolphin_logger.cli.shutil.copy')
@patch('builtins.print')
def test_handle_init_command_new_config(
    mock_print, mock_shutil_copy, mock_pkg_files, 
    mock_get_config_path, mock_get_config_dir, tmp_path
):
    mock_config_dir_path = tmp_path / ".dolphin-logger"
    mock_config_file_path = mock_config_dir_path / "config.json"

    mock_get_config_dir.return_value = mock_config_dir_path
    mock_get_config_path.return_value = mock_config_file_path
    
    # Simulate config file does not exist yet
    # Path.exists() will be called on mock_config_file_path
    with patch.object(Path, 'exists', return_value=False):
        # Mock for importlib.resources.files(...).joinpath(...).is_file()
        mock_traversable = MagicMock()
        mock_traversable.is_file.return_value = True
        mock_pkg_files.return_value.joinpath.return_value = mock_traversable

        _handle_init_command()

    mock_get_config_dir.assert_called_once()
    mock_get_config_path.assert_called_once()
    mock_pkg_files.assert_called_once_with('dolphin_logger')
    mock_pkg_files.return_value.joinpath.assert_called_once_with('config.json.example')
    mock_traversable.is_file.assert_called_once()
    
    # Ensure config_dir.mkdir was called (it's called by get_config_dir, but also in _handle_init_command)
    # The mock_config_dir_path itself does not have mkdir patched here.
    # We need to check if Path(mock_config_dir_path).mkdir was called.
    # This is tricky. Let's assume get_config_dir handles its own dir creation.
    # The _handle_init_command has a direct call too.
    # For simplicity, we'll trust get_config_dir works (tested in test_config.py)
    # and check the shutil.copy call.
    
    mock_shutil_copy.assert_called_once_with(str(mock_traversable), mock_config_file_path)
    mock_print.assert_any_call(f"Default configuration file created at: {mock_config_file_path}")


@patch('dolphin_logger.cli.get_config_dir')
@patch('dolphin_logger.cli.get_config_path')
@patch('dolphin_logger.cli.shutil.copy') # Should not be called
@patch('builtins.print')
def test_handle_init_command_config_exists(
    mock_print, mock_shutil_copy, mock_get_config_path, mock_get_config_dir, tmp_path
):
    mock_config_dir_path = tmp_path / ".dolphin-logger-exists"
    mock_config_file_path = mock_config_dir_path / "config.json"

    mock_get_config_dir.return_value = mock_config_dir_path
    mock_get_config_path.return_value = mock_config_file_path
    
    with patch.object(Path, 'exists', return_value=True): # Simulate config file *does* exist
        _handle_init_command()

    mock_shutil_copy.assert_not_called()
    mock_print.assert_any_call(f"Configuration file already exists at: {mock_config_file_path}")


@patch('dolphin_logger.cli.get_config_dir')
@patch('dolphin_logger.cli.get_config_path')
@patch('dolphin_logger.cli.pkg_resources.files')
@patch('builtins.print')
def test_handle_init_command_template_not_found(
    mock_print, mock_pkg_files, mock_get_config_path, mock_get_config_dir, tmp_path
):
    mock_config_dir_path = tmp_path / ".dolphin-logger-no-template"
    mock_config_file_path = mock_config_dir_path / "config.json"
    mock_get_config_dir.return_value = mock_config_dir_path
    mock_get_config_path.return_value = mock_config_file_path

    with patch.object(Path, 'exists', return_value=False): # Config doesn't exist
        mock_traversable = MagicMock()
        mock_traversable.is_file.return_value = False # Template not found
        mock_pkg_files.return_value.joinpath.return_value = mock_traversable
        
        _handle_init_command()
    
    mock_print.assert_any_call("Error: Default configuration template (config.json.example) not found within the package.")


# --- Tests for `config` command handler (_handle_config_command) ---
@patch('dolphin_logger.cli.get_config_path')
@patch('builtins.print')
def test_handle_config_command_path(mock_print, mock_get_config_path, tmp_path):
    expected_path = tmp_path / "test_cfg.json"
    mock_get_config_path.return_value = expected_path
    
    args = argparse.Namespace(path=True, validate=False)
    _handle_config_command(args)
    
    mock_print.assert_any_call(f"Expected configuration file path: {expected_path}")

@patch('dolphin_logger.cli.get_config_path')
@patch('dolphin_logger.cli.load_config') # Mock load_config from .config module
@patch('builtins.print')
def test_handle_config_command_validate_success(
    mock_print, mock_load_config_cli, mock_get_config_path_cli, tmp_path
):
    config_file = tmp_path / "valid_config.json"
    mock_get_config_path_cli.return_value = config_file
    
    # Simulate config file exists
    with patch.object(Path, 'exists', return_value=True):
        mock_load_config_cli.return_value = {"models": [{"id": "test"}]} # Simulate successful load
        args = argparse.Namespace(path=False, validate=True)
        _handle_config_command(args)

    mock_load_config_cli.assert_called_once()
    mock_print.assert_any_call("Configuration appears valid. 1 model(s) entries found.")


@patch('dolphin_logger.cli.get_config_path')
@patch('builtins.print')
def test_handle_config_command_validate_file_not_found(
    mock_print, mock_get_config_path_cli, tmp_path
):
    config_file = tmp_path / "non_existent_config.json"
    mock_get_config_path_cli.return_value = config_file
    
    with patch.object(Path, 'exists', return_value=False): # Simulate file NOT found
        args = argparse.Namespace(path=False, validate=True)
        _handle_config_command(args)
    
    mock_print.assert_any_call(f"Configuration file not found at: {config_file}")

@patch('dolphin_logger.cli.get_config_path')
@patch('dolphin_logger.cli.load_config', side_effect=json.JSONDecodeError("Syntax error", "doc", 0))
@patch('builtins.print')
def test_handle_config_command_validate_invalid_json(
    mock_print, mock_load_config_json_error, mock_get_config_path_cli, tmp_path
):
    config_file = tmp_path / "invalid_config.json"
    mock_get_config_path_cli.return_value = config_file
    
    with patch.object(Path, 'exists', return_value=True): # File exists
        args = argparse.Namespace(path=False, validate=True)
        _handle_config_command(args)

    mock_load_config_json_error.assert_called_once()
    mock_print.assert_any_call("Configuration validation failed: Invalid JSON - Syntax error (line 1, column 1)")


@patch('builtins.print')
def test_handle_config_command_no_flags(mock_print):
    args = argparse.Namespace(path=False, validate=False) # No flags set
    _handle_config_command(args)
    mock_print.assert_any_call("Please specify an option for the 'config' command: --path or --validate.")

# Test the initial load_config call in main_cli for server/upload commands
@patch('argparse.ArgumentParser.parse_args')
@patch('dolphin_logger.cli.load_config')
@patch('dolphin_logger.cli.run_server_main') # To prevent actual server start
def test_main_cli_initial_load_config_for_server(mock_run_server, mock_load_config, mock_parse_args):
    mock_parse_args.return_value = argparse.Namespace(command='server') # or None for default
    main_cli()
    mock_load_config.assert_called_once() # Should be called for server command
    mock_run_server.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('dolphin_logger.cli.load_config')
@patch('dolphin_logger.cli.upload_logs') # To prevent actual upload
def test_main_cli_initial_load_config_for_upload(mock_upload_logs, mock_load_config, mock_parse_args):
    mock_parse_args.return_value = argparse.Namespace(command='upload')
    main_cli()
    mock_load_config.assert_called_once() # Should be called for upload command
    mock_upload_logs.assert_called_once()

@patch('argparse.ArgumentParser.parse_args')
@patch('dolphin_logger.cli.load_config', side_effect=Exception("Config init failed severely"))
@patch('builtins.print')
def test_main_cli_initial_load_config_fails(mock_print, mock_load_config_severe_fail, mock_parse_args):
    # Test when the initial load_config for server/upload itself fails
    mock_parse_args.return_value = argparse.Namespace(command='server')
    main_cli()
    mock_load_config_severe_fail.assert_called_once()
    mock_print.assert_any_call("Error during configuration initialization for server: Config init failed severely")
    mock_print.assert_any_call("Please check your setup. Exiting.")

```
