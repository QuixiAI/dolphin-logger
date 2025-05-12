import os
import glob
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from datetime import datetime

DATASET_REPO_ID = "cognitivecomputations/dolphin-logger"

def find_jsonl_files():
    """Finds all .jsonl files in the current directory."""
    return glob.glob("*.jsonl")

def upload_and_create_pr():
    """Uploads .jsonl files to Hugging Face Hub and creates a PR."""
    api = HfApi()

    jsonl_files = find_jsonl_files()
    if not jsonl_files:
        print("No .jsonl files found to upload.")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s): {jsonl_files}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"upload-logs-{timestamp}"
    
    try:
        try:
            repo_info = api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset")
            print(f"Creating branch '{branch_name}' in dataset '{DATASET_REPO_ID}'")
            api.create_branch(repo_id=DATASET_REPO_ID, repo_type="dataset", branch=branch_name)
        except Exception as e:
            print(f"Could not create branch '{branch_name}': {e}")
            print("Attempting to upload to main branch directly. This is not recommended for PRs.")
            print(f"Failed to create branch '{branch_name}'. Aborting PR creation.")
            return


        commit_message = f"Add new log files: {', '.join(jsonl_files)}"
        
        operations = []
        for file_path in jsonl_files:
            path_in_repo = os.path.basename(file_path) # Use filename as path in repo
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=file_path,
                )
            )
            print(f"Prepared upload for {file_path} to {path_in_repo} on branch {branch_name}")

        if not operations:
            print("No files prepared for commit. This shouldn't happen if files were found.")
            return

        print(f"Creating commit on branch '{branch_name}' with message: '{commit_message}'")
        commit_info = api.create_commit(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            create_pr=True,
        )
        print(f"Successfully committed files to branch '{branch_name}' and created a Pull Request.")

        if hasattr(commit_info, "pr_url") and commit_info.pr_url:
            print(f"Successfully created Pull Request (Draft): {commit_info.pr_url}")
            print("Please review and merge the PR on Hugging Face Hub.")
        else:
            print("Pull Request was not created or PR URL not available.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    upload_and_create_pr()
