import pytest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
from pathlib import Path
from datetime import date, datetime, timedelta
import uuid
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions to be tested from the new logging_utils module
from dolphin_logger.logging_utils import get_current_log_file, _should_log_request
# Import other necessary components that might be used by these functions or their mocks
from dolphin_logger import logging_utils as dolphin_logging_module # To reset globals

# Fixture to reset logging globals before each test
@pytest.fixture(autouse=True)
def reset_logging_globals():
    """Reset globals in logging_utils.py before each test."""
    dolphin_logging_module.current_logfile_name = None
    dolphin_logging_module.current_logfile_date = None

# --- Tests for _should_log_request ---
@pytest.mark.parametrize("request_data, expected_log_decision", [
    ({"messages": [{"role": "user", "content": "Log this"}]}, True),
    ({"messages": [{"role": "user", "content": "### Task: Do not log this"}]}, False),
    ({"messages": [{"role": "user", "content": "  ### Task: Still no log"}]}, False),
    ({"messages": [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Log this user message"}]}, True),
    ({"messages": [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "### Task: No log for this one"}]}, False),
    ({"messages": []}, True), 
    ({"foo": "bar"}, True),   
    (None, True),             
    ({"messages": [{"role": "assistant", "content": "Assistant says something"}]}, True), 
    ({"messages": [{"role": "user", "content": ""}]}, True), 
    ({"messages": [{"role": "user", "content": "### Task"}]}, False), 
    ({"messages": [{"role": "user", "content": "Not a task ### Task: but contains it"}]}, True),
])
def test_should_log_request_logging(request_data, expected_log_decision, mocker):
    mocker.patch('builtins.print') # Suppress print from the function
    assert _should_log_request(request_data) == expected_log_decision

# --- Tests for get_current_log_file ---
# These tests are adapted from the original test_main.py
# They need to mock dependencies of get_current_log_file, which now includes
# get_logs_dir from dolphin_logger.config
@patch('dolphin_logger.logging_utils.uuid.uuid4') # Patch uuid where it's used
@patch('dolphin_logger.logging_utils.date') # Patch date where it's used
@patch('os.listdir') # Patch os.listdir globally as it's used by the function
@patch('os.path.getmtime') # Patch os.path.getmtime globally
@patch('dolphin_logger.config.get_logs_dir') # Patch the imported get_logs_dir
def test_get_current_log_file_creation_and_resume_logic_logging(
    mock_get_logs_dir_config, mock_os_path_getmtime, mock_os_listdir, 
    mock_date_logging, mock_uuid_logging, tmp_path # tmp_path for a temporary logs directory
):
    # Setup common mock returns
    mock_logs_dir_path = tmp_path / "test_logs_logging_utils"
    mock_logs_dir_path.mkdir(exist_ok=True)
    mock_get_logs_dir_config.return_value = mock_logs_dir_path # get_logs_dir returns this path
    
    today_date = date(2023, 10, 5)
    mock_date_logging.today.return_value = today_date
    
    # --- Test 1: No existing files, creates new ---
    # Reset logging globals explicitly here for clarity, though fixture does it
    dolphin_logging_module.current_logfile_name = None 
    dolphin_logging_module.current_logfile_date = None
    
    new_uuid_val = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    mock_uuid_logging.return_value = new_uuid_val
    mock_os_listdir.return_value = [] # No files in the directory
    
    log_file = get_current_log_file()
    expected_new_path = mock_logs_dir_path / f"{new_uuid_val}.jsonl"
    assert log_file == expected_new_path
    assert dolphin_logging_module.current_logfile_name == expected_new_path
    assert dolphin_logging_module.current_logfile_date == today_date
    mock_uuid_logging.assert_called_once()

    # --- Test 2: Existing file from today, should resume (latest one) ---
    dolphin_logging_module.current_logfile_name = None 
    dolphin_logging_module.current_logfile_date = None
    mock_uuid_logging.reset_mock() 

    exist_uuid1_str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    exist_uuid2_str = "cccccccc-cccc-cccc-cccc-cccccccccccc" # This one will be "later"
    mock_os_listdir.return_value = [f"{exist_uuid1_str}.jsonl", f"{exist_uuid2_str}.jsonl", "not-a-uuid.txt"]
    
    time_early = datetime(today_date.year, today_date.month, today_date.day, 9, 0, 0).timestamp()
    time_later = datetime(today_date.year, today_date.month, today_date.day, 10, 0, 0).timestamp()

    def getmtime_side_effect_resume(p_str): # os.path.getmtime takes string or PathLike
        p = Path(p_str) # Ensure we are working with Path objects for comparison
        if p.name == f"{exist_uuid1_str}.jsonl": return time_early
        if p.name == f"{exist_uuid2_str}.jsonl": return time_later
        return 0.0
    mock_os_path_getmtime.side_effect = getmtime_side_effect_resume
    
    log_file = get_current_log_file()
    expected_resume_path = mock_logs_dir_path / f"{exist_uuid2_str}.jsonl"
    assert log_file == expected_resume_path
    assert dolphin_logging_module.current_logfile_name == expected_resume_path
    assert dolphin_logging_module.current_logfile_date == today_date
    mock_uuid_logging.assert_not_called() 

    # --- Test 3: Date changes, should create new file ---
    # Globals are now set from the previous part of this test
    # dolphin_logging_module.current_logfile_name == expected_resume_path
    # dolphin_logging_module.current_logfile_date == today_date
    
    new_day_date = date(2023, 10, 6) 
    mock_date_logging.today.return_value = new_day_date # Date changes to next day
    
    new_day_uuid_val = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    mock_uuid_logging.return_value = new_day_uuid_val 
    mock_os_listdir.return_value = [] # No files for the new day yet

    log_file = get_current_log_file()
    expected_new_day_path = mock_logs_dir_path / f"{new_day_uuid_val}.jsonl"
    assert log_file == expected_new_day_path
    assert dolphin_logging_module.current_logfile_name == expected_new_day_path
    assert dolphin_logging_module.current_logfile_date == new_day_date
    mock_uuid_logging.assert_called_once() # Called for the new day's file

@patch('dolphin_logger.config.get_logs_dir') # Patch where get_logs_dir is imported from
@patch('dolphin_logger.logging_utils.date')   # Patch date where it's used
@patch('os.listdir')
@patch('os.path.getmtime', side_effect=OSError("Mocked Permission denied"))
def test_get_current_log_file_handles_oserror_getmtime_logging(
    mock_os_path_getmtime_error, mock_os_listdir, mock_date_logging, 
    mock_get_logs_dir_config, tmp_path, mocker # mocker for uuid.uuid4
):
    dolphin_logging_module.current_logfile_name = None
    dolphin_logging_module.current_logfile_date = None

    mock_logs_dir_path = tmp_path / "logs_oserror_logging"
    mock_logs_dir_path.mkdir()
    mock_get_logs_dir_config.return_value = mock_logs_dir_path

    fixed_date = date(2023, 1, 20)
    mock_date_logging.today.return_value = fixed_date
    
    # Simulate a file that will cause OSError on getmtime
    # Ensure it's a valid UUID name to pass initial checks
    problematic_uuid_str = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"
    mock_os_listdir.return_value = [f"{problematic_uuid_str}.jsonl"]
    
    new_uuid_val = uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    mocker.patch('dolphin_logger.logging_utils.uuid.uuid4', return_value=new_uuid_val)

    # Mock print to check for "Resuming log file" or "Creating new log file" messages
    with patch('builtins.print') as mock_print:
        log_file = get_current_log_file()
    
    expected_path = mock_logs_dir_path / f"{new_uuid_val}.jsonl"
    assert log_file == expected_path
    
    # getmtime was called for problematic_file.jsonl, then it should create a new one.
    # The path passed to getmtime will be absolute.
    mock_os_path_getmtime_error.assert_called_once_with(mock_logs_dir_path / f"{problematic_uuid_str}.jsonl")
    
    assert dolphin_logging_module.current_logfile_name == expected_path
    assert dolphin_logging_module.current_logfile_date == fixed_date
    
    # Verify new file creation message was printed
    printed_new_log = False
    for call_args in mock_print.call_args_list:
        if f"Creating new log file: {expected_path}" in call_args[0][0]:
            printed_new_log = True
            break
    assert printed_new_log, "Did not print message for creating new log file after OSError."

@patch('dolphin_logger.config.get_logs_dir')
@patch('dolphin_logger.logging_utils.date')
@patch('os.listdir', side_effect=OSError("Mocked listdir failed"))
def test_get_current_log_file_handles_oserror_listdir_logging(
    mock_os_listdir_error, mock_date_logging, mock_get_logs_dir_config, tmp_path, mocker
):
    dolphin_logging_module.current_logfile_name = None
    dolphin_logging_module.current_logfile_date = None

    mock_logs_dir_path = tmp_path / "logs_oserror_listdir"
    # Do not create it, or make it unreadable if that's the test.
    # Here, listdir itself fails, so path existence might not matter as much as the error.
    mock_get_logs_dir_config.return_value = mock_logs_dir_path 

    fixed_date = date(2023, 1, 21)
    mock_date_logging.today.return_value = fixed_date
        
    new_uuid_val = uuid.UUID("gggggggg-gggg-gggg-gggg-gggggggggggg")
    mocker.patch('dolphin_logger.logging_utils.uuid.uuid4', return_value=new_uuid_val)

    with patch('builtins.print') as mock_print:
        log_file = get_current_log_file()

    expected_path = mock_logs_dir_path / f"{new_uuid_val}.jsonl"
    assert log_file == expected_path # Should fall back to creating a new file
    
    # listdir was called and raised an error
    mock_os_listdir_error.assert_called_once_with(mock_logs_dir_path)
    
    # Check that a warning about logs_dir not being found or listdir error was printed by SUT (if implemented)
    # And that it then proceeded to create a new log file.
    printed_new_log = False
    for call_args in mock_print.call_args_list:
        if f"Creating new log file: {expected_path}" in call_args[0][0]:
            printed_new_log = True
            break
    assert printed_new_log, "Did not print message for creating new log file after listdir OSError."

```
