import os
import json
import uuid
from datetime import date, datetime
from threading import Lock
from pathlib import Path

# Assuming get_logs_dir will be imported from .config in files that use these utils
# For example, in core_proxy.py or server.py where actual logging to file happens.
# This file itself doesn't need get_logs_dir directly if it only defines these.
# However, get_current_log_file *does* use it. So, it needs the import.
from .config import get_logs_dir

log_lock = Lock()

# Globals for daily log file management
current_logfile_name: Path | None = None
current_logfile_date: date | None = None

def get_current_log_file() -> Path:
    """
    Determines the current log file path.
    Creates a new log file if it's a new day or if no log file exists for today.
    Resumes the latest log file if one exists for today.
    """
    global current_logfile_name, current_logfile_date
    today = date.today()
    logs_dir = get_logs_dir()

    with log_lock:
        if current_logfile_name is None or current_logfile_date != today:
            # Reset or date changed, try to find or create a new one for today
            latest_log_file_today = None
            latest_mod_time = 0.0

            if current_logfile_date != today and current_logfile_name is not None:
                 print(f"Date changed. Old log file: {current_logfile_name} for date: {current_logfile_date}")


            # Scan the logs directory for files from today
            # This part is crucial if the process restarts on the same day.
            if os.path.exists(logs_dir): # Ensure logs_dir exists before listing
                for item_name in os.listdir(logs_dir):
                    if item_name.endswith(".jsonl"):
                        try:
                            uuid_part = item_name[:-6]
                            uuid.UUID(uuid_part) 
                        except ValueError:
                            continue 

                        filepath = logs_dir / item_name
                        try:
                            mod_time_timestamp = os.path.getmtime(filepath)
                            mod_date_obj = datetime.fromtimestamp(mod_time_timestamp).date()

                            if mod_date_obj == today:
                                if mod_time_timestamp > latest_mod_time:
                                    latest_mod_time = mod_time_timestamp
                                    latest_log_file_today = item_name
                        except OSError:
                            continue
            else: # logs_dir might not exist if get_logs_dir() failed or was mocked in a weird way
                print(f"Warning: Logs directory {logs_dir} not found during log file scan.")
                # Fallback to creating logs_dir, though get_logs_dir should handle this.
                try:
                    logs_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"Critical Error: Could not create logs directory {logs_dir}. Logging will likely fail. Error: {e}")
                    # Potentially raise an error or use a very temporary fallback in /tmp if this is critical
                    # For now, proceed and let file open fail later if dir is not writable.


            if latest_log_file_today:
                current_logfile_name = logs_dir / latest_log_file_today
                print(f"Resuming log file: {current_logfile_name} for date: {today}")
            else:
                new_uuid = uuid.uuid4()
                new_logfile_path = logs_dir / f"{new_uuid}.jsonl"
                current_logfile_name = new_logfile_path
                print(f"Creating new log file: {current_logfile_name} for date: {today}")
            
            current_logfile_date = today

    if current_logfile_name is None:
        # Fallback if something went extremely wrong, to prevent returning None
        # This should ideally never be reached if logs_dir is writable.
        fallback_uuid = uuid.uuid4()
        current_logfile_name = get_logs_dir() / f"fallback_{fallback_uuid}.jsonl" # Use get_logs_dir again to ensure it's the correct path
        current_logfile_date = today # Should be set with today's date
        print(f"CRITICAL FALLBACK: Using emergency log file: {current_logfile_name}")


    return current_logfile_name

def _should_log_request(current_request_data: dict | None) -> bool:
    """
    Determines if a request should be logged based on its content.
    Does not log if the first 'user' message content starts with '### Task'.
    """
    if not current_request_data or 'messages' not in current_request_data or not isinstance(current_request_data['messages'], list):
        return True

    for message in current_request_data['messages']:
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content')
            if isinstance(content, str) and content.strip().startswith("### Task"):
                print("Skipping log for request starting with '### Task'")
                return False
            return True 
    return True
