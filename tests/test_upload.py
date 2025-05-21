import pytest
from unittest.mock import patch, MagicMock, call
import os
from pathlib import Path
from datetime import datetime
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions to be tested from the new upload module
from dolphin_logger.upload import find_jsonl_files, upload_logs, DATASET_REPO_ID
# Import other necessary components
from dolphin_logger.config import get_logs_dir # upload_logs uses this

# --- Tests for find_jsonl_files ---
@patch('dolphin_logger.upload.glob.glob') # Patch glob where it's used in upload.py
@patch('dolphin_logger.upload.get_logs_dir') # Mock get_logs_dir if find_jsonl_files uses it directly
                                         # Current find_jsonl_files takes log_dir as param.
def test_find_jsonl_files_upload(mock_get_logs_dir_not_used_here, mock_glob_glob):
    mock_logs_dir_path_str = "/mock/logs_upload_test"
    # mock_get_logs_dir.return_value = Path(mock_logs_dir_path_str) # Not needed if log_dir is a param
    
    expected_files = ["/mock/logs_upload_test/log1.jsonl", "/mock/logs_upload_test/log2.jsonl"]
    mock_glob_glob.return_value = expected_files

    # Pass a string path, as os.path.join is used inside
    result = find_jsonl_files(mock_logs_dir_path_str)
    
    assert result == expected_files
    mock_glob_glob.assert_called_once_with(os.path.join(mock_logs_dir_path_str, "*.jsonl"))

# --- Tests for upload_logs ---
# These tests are more integration-like for the upload_logs function itself,
# focusing on its interaction with HfApi and file system utilities.

@patch('dolphin_logger.upload.HfApi')
@patch('dolphin_logger.upload.find_jsonl_files')
@patch('dolphin_logger.upload.get_logs_dir') # Patched where it's imported in upload.py
def test_upload_logs_success_upload(
    mock_get_logs_dir_upload, mock_find_jsonl_upload, mock_hf_api_class_upload
):
    mock_logs_dir_path = Path("/mock/logs_for_hf_upload")
    mock_get_logs_dir_upload.return_value = mock_logs_dir_path
    
    # Simulate some .jsonl files found
    mock_found_files = ["/mock/logs_for_hf_upload/logA.jsonl", "/mock/logs_for_hf_upload/logB.jsonl"]
    mock_find_jsonl_upload.return_value = mock_found_files

    mock_hf_api_instance = MagicMock()
    mock_hf_api_class_upload.return_value = mock_hf_api_instance
    
    # Simulate repo_info and create_branch succeeding
    mock_hf_api_instance.repo_info.return_value = MagicMock(id="dataset_id") # Dummy repo info
    
    # Simulate create_commit succeeding and returning an object with pr_url
    commit_info_mock = MagicMock()
    commit_info_mock.pr_url = "http://hf.co/datasets/cognitivecomputations/dolphin-logger/pull/123"
    mock_hf_api_instance.create_commit.return_value = commit_info_mock

    # Mock print to check output messages
    with patch('builtins.print') as mock_print:
        upload_logs()

    mock_get_logs_dir_upload.assert_called_once()
    mock_find_jsonl_upload.assert_called_once_with(mock_logs_dir_path)
    mock_hf_api_class_upload.assert_called_once() # HfApi instantiated

    mock_hf_api_instance.repo_info.assert_called_once_with(repo_id=DATASET_REPO_ID, repo_type="dataset")
    mock_hf_api_instance.create_branch.assert_called_once() # Branch name is dynamic with timestamp
    
    # Check create_commit call
    assert mock_hf_api_instance.create_commit.call_count == 1
    args, kwargs = mock_hf_api_instance.create_commit.call_args
    assert kwargs['repo_id'] == DATASET_REPO_ID
    assert kwargs['repo_type'] == "dataset"
    assert kwargs['create_pr'] == True
    assert len(kwargs['operations']) == 2 # Two files
    assert kwargs['operations'][0].path_in_repo == "logA.jsonl" # Basename
    assert kwargs['operations'][0].path_or_fileobj == "/mock/logs_for_hf_upload/logA.jsonl"
    assert kwargs['operations'][1].path_in_repo == "logB.jsonl"
    assert kwargs['operations'][1].path_or_fileobj == "/mock/logs_for_hf_upload/logB.jsonl"
    
    # Check for PR URL print
    printed_pr_url = False
    for call_arg in mock_print.call_args_list:
        if "Pull Request (Draft): http://hf.co/datasets/cognitivecomputations/dolphin-logger/pull/123" in call_arg[0][0]:
            printed_pr_url = True
            break
    assert printed_pr_url, "PR URL was not printed on successful upload."

@patch('dolphin_logger.upload.HfApi')
@patch('dolphin_logger.upload.find_jsonl_files')
@patch('dolphin_logger.upload.get_logs_dir')
def test_upload_logs_no_files_to_upload(
    mock_get_logs_dir_upload, mock_find_jsonl_upload, mock_hf_api_class_upload
):
    mock_logs_dir_path = Path("/mock/logs_empty_for_hf")
    mock_get_logs_dir_upload.return_value = mock_logs_dir_path
    mock_find_jsonl_upload.return_value = [] # No files found

    with patch('builtins.print') as mock_print:
        upload_logs()
    
    mock_hf_api_class_upload.assert_not_called() # HfApi should not be instantiated
    mock_get_logs_dir_upload.assert_called_once()
    mock_find_jsonl_upload.assert_called_once_with(mock_logs_dir_path)
    
    no_files_msg_found = False
    for call_arg in mock_print.call_args_list:
        if f"No .jsonl files found in {mock_logs_dir_path} to upload." in call_arg[0][0]:
            no_files_msg_found = True
            break
    assert no_files_msg_found, "Message for no files found was not printed."

@patch('dolphin_logger.upload.HfApi')
@patch('dolphin_logger.upload.find_jsonl_files')
@patch('dolphin_logger.upload.get_logs_dir')
def test_upload_logs_branch_creation_fails_upload(
    mock_get_logs_dir_upload, mock_find_jsonl_upload, mock_hf_api_class_upload
):
    mock_logs_dir_path = Path("/mock/logs_branch_fail_hf")
    mock_get_logs_dir_upload.return_value = mock_logs_dir_path
    mock_find_jsonl_upload.return_value = ["/mock/logs_branch_fail_hf/log1.jsonl"] # One file to attempt

    mock_hf_api_instance = MagicMock()
    mock_hf_api_class_upload.return_value = mock_hf_api_instance
    mock_hf_api_instance.repo_info.return_value = MagicMock() # Repo info succeeds
    # Simulate create_branch failing
    mock_hf_api_instance.create_branch.side_effect = Exception("Mocked HfApi branch creation failure")

    with patch('builtins.print') as mock_print:
        upload_logs()

    mock_hf_api_instance.create_commit.assert_not_called() # Should not attempt commit if branch fails
    
    branch_fail_msg_found = False
    for call_arg in mock_print.call_args_list:
        if "Failed to create branch" in call_arg[0][0] and "Aborting PR creation" in call_arg[0][0]:
            branch_fail_msg_found = True
            break
    assert branch_fail_msg_found, "Message for branch creation failure was not printed."

@patch('dolphin_logger.upload.HfApi')
@patch('dolphin_logger.upload.find_jsonl_files')
@patch('dolphin_logger.upload.get_logs_dir')
def test_upload_logs_commit_fails_upload(
    mock_get_logs_dir_upload, mock_find_jsonl_upload, mock_hf_api_class_upload
):
    mock_logs_dir_path = Path("/mock/logs_commit_fail_hf")
    mock_get_logs_dir_upload.return_value = mock_logs_dir_path
    mock_find_jsonl_upload.return_value = ["/mock/logs_commit_fail_hf/log1.jsonl"]

    mock_hf_api_instance = MagicMock()
    mock_hf_api_class_upload.return_value = mock_hf_api_instance
    mock_hf_api_instance.repo_info.return_value = MagicMock()
    # create_branch succeeds
    mock_hf_api_instance.create_branch.return_value = None 
    # create_commit fails
    mock_hf_api_instance.create_commit.side_effect = Exception("Mocked HfApi commit failure")

    with patch('builtins.print') as mock_print:
        upload_logs()
    
    # Check that an error message related to the commit failure was printed
    commit_fail_msg_found = False
    for call_arg in mock_print.call_args_list:
        # The generic error message from the except block in upload_logs
        if "An error occurred during upload: Mocked HfApi commit failure" in call_arg[0][0]:
            commit_fail_msg_found = True
            break
    assert commit_fail_msg_found, "Message for commit failure was not printed."

```
