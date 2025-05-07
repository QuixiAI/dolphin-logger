import os
import glob
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd
from datetime import datetime

# Configuration
DATASET_REPO_ID = "cognitivecomputations/coding-collect"
# Ensure HF_TOKEN is set in your environment variables
# HfFolder.save_token(os.getenv("HF_TOKEN")) # Not strictly needed if token is already globally configured or passed to HfApi

def find_jsonl_files():
    """Finds all .jsonl files in the current directory."""
    return glob.glob("*.jsonl")

def upload_and_create_pr():
    """Uploads .jsonl files to Hugging Face Hub and creates a PR."""
    # Assuming the user is already logged in via `huggingface-cli login`
    api = HfApi()

    jsonl_files = find_jsonl_files()
    if not jsonl_files:
        print("No .jsonl files found to upload.")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s): {jsonl_files}")

    # Create a unique branch name for the PR
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"upload-logs-{timestamp}"
    
    try:
        # Check if repo exists and get default branch
        try:
            repo_info = api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset")
            # default_branch = repo_info.sha # This is the main branch commit hash, not name
            # For datasets, 'main' is typically the default branch.
            # If not, one might need to fetch this differently or assume 'main'.
            # Creating a branch from default usually works without specifying base.
            print(f"Creating branch '{branch_name}' in dataset '{DATASET_REPO_ID}'")
            api.create_branch(repo_id=DATASET_REPO_ID, repo_type="dataset", branch=branch_name)
        except Exception as e:
            # Fallback or error if branch creation fails (e.g. branch already exists, though unlikely with timestamp)
            print(f"Could not create branch '{branch_name}': {e}")
            print("Attempting to upload to main branch directly. This is not recommended for PRs.")
            # If direct upload to main is desired, the PR step would be skipped or handled differently.
            # For now, we'll proceed assuming branch creation is key for a PR.
            # If branch creation fails, we might want to stop or have a different flow.
            # For simplicity, if branch creation fails, we'll try to commit to it anyway,
            # or one could choose to commit to 'main' and skip PR.
            # However, create_commit requires a branch.
            # Let's assume if create_branch fails, we should stop for a PR workflow.
            print(f"Failed to create branch '{branch_name}'. Aborting PR creation.")
            # As an alternative, one could try to upload to 'main' and then manually create a PR
            # or skip the PR if direct commits are acceptable.
            # For this script, we'll stick to the PR workflow.
            return


        commit_message = f"Add new log files: {', '.join(jsonl_files)}"
        
        operations = []
        for file_path in jsonl_files:
            # The path_in_repo should ideally place files in a structured way,
            # e.g., under a 'data/' folder or similar, if the dataset expects it.
            # For now, uploading to the root of the branch.
            # If the dataset has a specific structure (e.g. 'data/train.jsonl'), adjust path_in_repo.
            # Assuming we upload them to the root of the dataset repo for now.
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
        api.create_commit(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
            revision=branch_name,
        )
        print(f"Successfully committed files to branch '{branch_name}'.")

        # Create a pull request
        pr_title = f"Upload new log data ({timestamp})"
        pr_description = f"This PR adds {len(jsonl_files)} new log file(s) collected on {datetime.now().strftime('%Y-%m-%d')}."
        
        print(f"Creating Pull Request from '{branch_name}' to main branch...")
        pull_request = api.create_pull_request(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            title=pr_title,
            description=pr_description
            # The head branch is implicitly the one set in `revision` of `create_commit`
            # The base branch is implicitly the default branch of the repository (e.g., 'main')
            # Programmatically created PRs are in "draft" status by default.
        )
        # Construct the PR URL manually
        # The pull_request object is an instance of DiscussionWithDetails
        # It has 'num' (the discussion/PR number) and 'repo_id' attributes.
        pr_url = f"https://huggingface.co/datasets/{pull_request.repo_id}/discussions/{pull_request.num}"
        print(f"Successfully created Pull Request (Draft): {pr_url}")
        print("Please review and merge the PR on Hugging Face Hub.")

        # Optionally, delete the local files after successful upload and PR creation
        # for file_path in jsonl_files:
        #     os.remove(file_path)
        #     print(f"Deleted local file: {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Consider cleaning up the created branch if something went wrong after its creation
        # try:
        #     api.delete_branch(repo_id=DATASET_REPO_ID, repo_type="dataset", branch=branch_name)
        #     print(f"Cleaned up branch {branch_name} due to error.")
        # except Exception as delete_e:
        #     print(f"Failed to clean up branch {branch_name}: {delete_e}")


if __name__ == "__main__":
    upload_and_create_pr()
