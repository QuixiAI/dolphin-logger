import os
import glob
from datetime import datetime
from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

# Import get_logs_dir and get_huggingface_repo from the new config.py
from .config import get_logs_dir, get_huggingface_repo

def find_jsonl_files(log_dir: str | Path) -> list[str]:
    """Finds all .jsonl files in the specified log directory."""
    return glob.glob(os.path.join(str(log_dir), "*.jsonl"))

def upload_logs():
    """Uploads .jsonl files from the log directory to Hugging Face Hub and creates a PR."""
    api = HfApi()
    # Use get_logs_dir from .config
    log_dir_path = get_logs_dir()
    # Get the configurable Hugging Face repository
    dataset_repo_id = get_huggingface_repo()
    jsonl_files = find_jsonl_files(log_dir_path)

    if not jsonl_files:
        print(f"No .jsonl files found in {log_dir_path} to upload.")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s) in {log_dir_path}: {jsonl_files}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"upload-logs-{timestamp}"

    try:
        try:
            repo_info = api.repo_info(repo_id=dataset_repo_id, repo_type="dataset")
            print(f"Creating branch '{branch_name}' in dataset '{dataset_repo_id}'")
            api.create_branch(repo_id=dataset_repo_id, repo_type="dataset", branch=branch_name)
        except Exception as e:
            print(f"Could not create branch '{branch_name}': {e}")
            print(f"Failed to create branch '{branch_name}'. Aborting PR creation.")
            return

        commit_message = f"Add new log files from {log_dir_path} ({timestamp})"

        operations = []
        for file_path_str in jsonl_files:
            path_in_repo = os.path.basename(file_path_str) 
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=file_path_str,
                )
            )
            print(f"Prepared upload for {file_path_str} to {path_in_repo} on branch {branch_name}")

        if not operations:
            print("No files prepared for commit. This shouldn't happen if files were found.")
            return

        print(f"Creating commit on branch '{branch_name}' with message: '{commit_message}'")
        commit_info = api.create_commit(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            create_pr=True,
        )
        print(f"Successfully committed files to branch '{branch_name}'.")

        if hasattr(commit_info, 'pr_url') and commit_info.pr_url:
            print(f"Successfully created Pull Request (Draft): {commit_info.pr_url}")
            print("Please review and merge the PR on Hugging Face Hub.")
        elif hasattr(commit_info, 'commit_url') and commit_info.commit_url:
            print(f"Commit successful: {commit_info.commit_url}")
            print("A Pull Request may have been created. Please check the repository on Hugging Face Hub.")
            if not getattr(commit_info, 'pr_url', None):
                 print("Note: create_pr was True, but no PR URL was returned. A PR might still exist or need manual creation.")
        else:
            print("Commit successful, but Pull Request URL not available. Please check and create PR manually if needed.")

    except Exception as e:
        print(f"An error occurred during upload: {e}")
        # import traceback
        # traceback.print_exc()
