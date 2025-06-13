import os
from pathlib import Path
import json
import shutil

def get_config_dir() -> Path:
    """
    Returns the path to the dolphin-logger configuration directory (~/.dolphin-logger).
    Creates the directory if it doesn't exist.
    """
    config_dir = Path.home() / ".dolphin-logger"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_path() -> Path:
    """Returns the path to the config.json file."""
    return get_config_dir() / "config.json"

def get_logs_dir() -> Path:
    """
    Returns the path to the directory where logs should be stored.
    Creates the directory if it doesn't exist.
    """
    logs_dir = get_config_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def load_config() -> dict:
    """
    Loads the configuration from ~/.dolphin-logger/config.json if it exists.
    If it doesn't exist, it copies the default config from the package
    to ~/.dolphin-logger/config.json and loads it.
    """
    user_config_path = get_config_path()
    package_config_path = Path(__file__).parent / "config.json"

    if not user_config_path.exists():
        # Copy the default config to the user's config directory
        shutil.copy(package_config_path, user_config_path)
        print(f"Copied default config to {user_config_path}")

    config_path = user_config_path
    config_data = {}

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        # This case is handled by the copy, but as a safeguard:
        raise FileNotFoundError(f"Config file not found at {config_path}, and default was not copied.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Error decoding JSON from config file at {config_path}", "", 0) # Added empty doc and pos for compatibility

    if 'models' in config_data and isinstance(config_data['models'], list):
        for model_entry in config_data['models']:
            if isinstance(model_entry, dict) and 'apiKey' in model_entry:
                api_key_value = model_entry['apiKey']
                if isinstance(api_key_value, str) and api_key_value.startswith("ENV:"):
                    env_var_name = api_key_value[4:]
                    env_var_value = os.environ.get(env_var_name)
                    if env_var_value:
                        model_entry['apiKey'] = env_var_value
                        print(f"Resolved API key for model '{model_entry.get('model', 'Unknown Model')}' from environment variable '{env_var_name}'.")
                    else:
                        model_entry['apiKey'] = None # Or ""
                        print(f"Warning: Environment variable '{env_var_name}' not found for model '{model_entry.get('model', 'Unknown Model')}'. API key set to None.")
    
    return config_data

def get_huggingface_repo() -> str:
    """
    Returns the Hugging Face repository from the configuration.
    If not specified in config, returns the default value.
    """
    config = load_config()
    return config.get("huggingface_repo", "cognitivecomputations/dolphin-logger")
