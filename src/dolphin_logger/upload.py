import os
import glob
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path

# Import get_logs_dir and get_huggingface_repo from the new config.py
from .config import get_logs_dir, get_huggingface_repo

# huggingface-hub is now a hard dependency, datasets is optional
from huggingface_hub import HfApi, CommitOperationAdd

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# PII and sensitive data patterns
PII_PATTERNS = [
    # API Keys and tokens
    (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', '[API_KEY_REDACTED]'),
    (r'(?i)(access[_-]?token|accesstoken)\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', '[ACCESS_TOKEN_REDACTED]'),
    (r'(?i)(secret[_-]?key|secretkey)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', '[SECRET_KEY_REDACTED]'),
    (r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{20,})', r'\1[BEARER_TOKEN_REDACTED]'),
    (r'(?i)(authorization:\s*bearer\s+)([a-zA-Z0-9_\-\.]{20,})', r'\1[AUTH_TOKEN_REDACTED]'),
    
    # OpenAI API keys
    (r'sk-[a-zA-Z0-9]{48}', '[OPENAI_API_KEY_REDACTED]'),
    
    # Anthropic API keys
    (r'sk-ant-[a-zA-Z0-9\-_]{95,}', '[ANTHROPIC_API_KEY_REDACTED]'),
    
    # Google API keys
    (r'AIza[0-9A-Za-z\-_]{35}', '[GOOGLE_API_KEY_REDACTED]'),
    
    # AWS keys
    (r'AKIA[0-9A-Z]{16}', '[AWS_ACCESS_KEY_REDACTED]'),
    (r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', '[AWS_SECRET_KEY_REDACTED]'),
    
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
    
    # Phone numbers (US format)
    (r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', '[PHONE_REDACTED]'),
    
    # Credit card numbers (basic pattern)
    (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b', '[CREDIT_CARD_REDACTED]'),
    
    # SSN (US format)
    (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]'),
    
    # Generic passwords
    (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{8,})["\']?', r'\1: [PASSWORD_REDACTED]'),
    
    # JWT tokens
    (r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*', '[JWT_TOKEN_REDACTED]'),
    
    # Generic tokens (long alphanumeric strings that might be sensitive)
    (r'(?i)(token|key|secret)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{32,})["\']?', r'\1: [TOKEN_REDACTED]'),
]

def sanitize_text(text: str) -> str:
    """Remove PII and sensitive data from text using regex patterns."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def filter_jsonl_file(input_file: str, output_file: str) -> int:
    """
    Filter PII and API keys from a JSONL file line by line.
    Returns the number of lines processed.
    """
    lines_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Convert to string, sanitize, then parse back
                json_str = json.dumps(data)
                sanitized_str = sanitize_text(json_str)
                sanitized_data = json.loads(sanitized_str)
                
                # Write sanitized line
                outfile.write(json.dumps(sanitized_data) + '\n')
                lines_processed += 1
                
            except json.JSONDecodeError:
                # If line is not valid JSON, sanitize as plain text
                sanitized_line = sanitize_text(line)
                outfile.write(sanitized_line + '\n')
                lines_processed += 1
    
    return lines_processed

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
    print("Filtering PII and API keys from log files...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"upload-logs-{timestamp}"

    # Create temporary directory for filtered files
    temp_dir = tempfile.mkdtemp(prefix="dolphin_logger_filtered_")
    filtered_files = []

    try:
        # Filter each JSONL file
        for file_path_str in jsonl_files:
            filename = os.path.basename(file_path_str)
            filtered_file_path = os.path.join(temp_dir, f"filtered_{filename}")
            
            print(f"Filtering {filename}...")
            lines_processed = filter_jsonl_file(file_path_str, filtered_file_path)
            print(f"Processed {lines_processed} lines from {filename}")
            
            filtered_files.append(filtered_file_path)

        try:
            repo_info = api.repo_info(repo_id=dataset_repo_id, repo_type="dataset")
            print(f"Creating branch '{branch_name}' in dataset '{dataset_repo_id}'")
            api.create_branch(repo_id=dataset_repo_id, repo_type="dataset", branch=branch_name)
        except Exception as e:
            print(f"Could not create branch '{branch_name}': {e}")
            print(f"Failed to create branch '{branch_name}'. Aborting PR creation.")
            return

        commit_message = f"Add filtered log files from {log_dir_path} ({timestamp}) - PII and API keys redacted"

        operations = []
        for filtered_file_path in filtered_files:
            # Use original filename (without "filtered_" prefix)
            original_filename = os.path.basename(filtered_file_path).replace("filtered_", "")
            path_in_repo = original_filename
            
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=filtered_file_path,
                )
            )
            print(f"Prepared upload for filtered {original_filename} to {path_in_repo} on branch {branch_name}")

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
        print(f"Successfully committed filtered files to branch '{branch_name}'.")

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
    finally:
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
