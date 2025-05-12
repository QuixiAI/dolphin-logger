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

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON in config file at {config_path}")
