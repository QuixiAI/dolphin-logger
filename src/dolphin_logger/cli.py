import argparse
import shutil
import sys
import importlib.resources as pkg_resources # Use importlib.resources
import os # For os.path.exists in init if needed, though Path.exists is better
from pathlib import Path # For Path object operations
import getpass
import subprocess
import requests
from typing import Dict, List, Optional

# Imports from our refactored modules
from .config import load_config, get_config_path, get_config_dir # Added get_config_dir
from .server import run_server_main
from .upload import upload_logs
import json # For validate in config command

def _interactive_api_key_setup():
    """Interactive API key setup with provider-specific instructions."""
    print("\n🔐 API Key Setup")
    print("=" * 50)
    
    providers_info = {
        "anthropic": {
            "name": "Anthropic (Claude)",
            "env_var": "ANTHROPIC_API_KEY", 
            "url": "https://console.anthropic.com/",
            "description": "Required for Claude models"
        },
        "openai": {
            "name": "OpenAI (GPT)",
            "env_var": "OPENAI_API_KEY",
            "url": "https://platform.openai.com/api-keys", 
            "description": "Required for GPT models"
        },
        "google": {
            "name": "Google AI (Gemini)",
            "env_var": "GOOGLE_API_KEY",
            "url": "https://makersuite.google.com/app/apikey",
            "description": "Required for Gemini models"
        }
    }
    
    configured_keys = []
    
    for provider_id, info in providers_info.items():
        print(f"\n📋 {info['name']}")
        print(f"   {info['description']}")
        print(f"   Get your API key: {info['url']}")
        
        current_key = os.environ.get(info['env_var'])
        if current_key:
            print(f"   ✅ {info['env_var']} is already set")
            configured_keys.append(provider_id)
            continue
            
        while True:
            choice = input(f"   Configure {info['name']} API key? (y/n/skip): ").lower().strip()
            if choice in ['y', 'yes']:
                key = getpass.getpass(f"   Enter your {info['name']} API key: ").strip()
                if key:
                    print(f"\n   Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
                    print(f"   export {info['env_var']}='{key}'")
                    print(f"   \n   Or run this command now:")
                    print(f"   export {info['env_var']}='{key}'")
                    configured_keys.append(provider_id)
                break
            elif choice in ['n', 'no', 'skip']:
                print(f"   ⏭️  Skipped {info['name']} - you can configure this later")
                break
            else:
                print("   Please enter 'y' for yes or 'n' for no")
    
    if configured_keys:
        print(f"\n🎉 Configured API keys for: {', '.join(configured_keys)}")
        print("\n💡 Pro tip: Add these export commands to your shell profile")
        print("   so they're available in future terminal sessions!")
    else:
        print("\n⚠️  No API keys configured. You can set them later using environment variables.")
    
    return configured_keys

def _test_configuration():
    """Test the current configuration by checking provider availability."""
    print("\n🧪 Testing Configuration")
    print("=" * 50)
    
    try:
        config_data = load_config()
        models = config_data.get('models', [])
        
        if not models:
            print("❌ No models configured")
            return False
            
        print(f"📊 Found {len(models)} configured models:")
        
        provider_status = {}
        for model in models:
            provider = model.get('provider', 'unknown')
            model_name = model.get('model', 'unknown')
            api_key = model.get('apiKey')
            
            # Check API key status
            key_status = "✅" if api_key and api_key != "None" else "❌"
            if api_key and api_key.startswith("ENV:"):
                env_var = api_key[4:]
                actual_key = os.environ.get(env_var)
                key_status = "✅" if actual_key else "❌"
                
            print(f"   • {model_name} ({provider}) - API Key: {key_status}")
            
            if provider not in provider_status:
                provider_status[provider] = {"total": 0, "configured": 0}
            provider_status[provider]["total"] += 1
            if key_status == "✅":
                provider_status[provider]["configured"] += 1
        
        print(f"\n📈 Provider Summary:")
        for provider, status in provider_status.items():
            configured = status["configured"]
            total = status["total"]
            status_icon = "✅" if configured == total else "⚠️" if configured > 0 else "❌"
            print(f"   {status_icon} {provider.title()}: {configured}/{total} models ready")
        
        return any(status["configured"] > 0 for status in provider_status.values())
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def _handle_init_command(interactive=False):
    """Handles the `dolphin-logger init` command."""
    print("🐬 Dolphin Logger Configuration Setup")
    print("=" * 50)
    
    config_dir = get_config_dir()
    config_file_path = get_config_path()

    if config_file_path.exists():
        print(f"📁 Configuration file already exists at: {config_file_path}")
        if interactive:
            choice = input("Do you want to reconfigure? (y/n): ").lower().strip()
            if choice not in ['y', 'yes']:
                print("Setup cancelled.")
                return
    else:
        try:
            # Use config.json.example from the project root as the template
            template_path = Path(__file__).parent.parent.parent / "config.json.example"
            
            if template_path.exists():
                config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(template_path, config_file_path)
                print(f"✅ Default configuration file created at: {config_file_path}")
            else:
                print(f"❌ Error: Default configuration template not found at {template_path}")
                print("Please ensure the package is installed correctly.")
                return

        except Exception as e:
            print(f"❌ Error during configuration initialization: {e}")
            return

    if interactive:
        # Interactive setup
        configured_keys = _interactive_api_key_setup()
        
        # Test configuration
        if _test_configuration():
            print("\n🎉 Configuration setup complete!")
        else:
            print("\n⚠️  Setup complete, but some issues detected.")
            print("You may need to configure API keys or check your configuration.")
            
        print(f"\n📋 Next Steps:")
        print(f"1. Start the server: dolphin-logger")
        print(f"2. Test with: curl http://localhost:5001/v1/models")
        print(f"3. Configure your LLM client to use: http://localhost:5001")
        
    else:
        # Standard setup
        print("\n🎉 Configuration initialized! Next steps:")
        print("1. Set your API keys as environment variables:")
        print("   export ANTHROPIC_API_KEY=your_anthropic_key")
        print("   export OPENAI_API_KEY=your_openai_key")
        print("   export GOOGLE_API_KEY=your_google_key")
        print("2. (Optional) Edit the config file to customize models and endpoints")
        print("3. Start the server: dolphin-logger")
        print("4. Test with: curl http://localhost:5001/v1/models")
        print("\n💡 For guided setup, use: dolphin-logger init --interactive")

def _handle_config_command(args):
    """Handles the `dolphin-logger config` command."""
    if args.path:
        config_file_path = get_config_path()
        print(config_file_path)
        if not config_file_path.exists():
             print(f"Note: Configuration file does not currently exist at this path. Run 'dolphin-logger init' to create it.", file=sys.stderr)
    
    elif args.validate:
        config_file_path = get_config_path()
        print(f"Validating configuration at: {config_file_path}...")
        if not config_file_path.exists():
            print(f"Configuration file not found at: {config_file_path}")
            print("Run 'dolphin-logger init' to create a default configuration file.")
            return

        try:
            # load_config already prints details about API key resolution
            config_data = load_config() 
            if config_data: # load_config returns a dict
                models_loaded = len(config_data.get("models", []))
                print(f"Configuration appears valid. {models_loaded} model(s) entries found.")
                # Further validation could be added here, e.g., checking schema
            else:
                # This case might occur if load_config returns None or empty dict on some error
                # though current load_config raises exceptions or returns dict.
                print("Configuration loaded but seems empty or invalid.")
        except json.JSONDecodeError as e:
            print(f"Configuration validation failed: Invalid JSON - {e.msg} (line {e.lineno}, column {e.colno})")
        except FileNotFoundError: # Should be caught by the .exists() check, but as a safeguard
            print(f"Configuration validation failed: File not found at {config_file_path}")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            # import traceback
            # traceback.print_exc()
    else:
        # No flags given for 'config' command, print help for this subcommand
        print("Please specify an option for the 'config' command: --path or --validate.")
        # Alternatively, could show the config path by default. For now, require a flag.


def main_cli():
    """Command Line Interface entry point for Dolphin Logger."""
    parser = argparse.ArgumentParser(description="Dolphin Logger: Proxy server, log uploader, and config manager.")
    
    # Add --port to the main parser so it works with default server behavior
    parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 5001, or PORT environment variable)')
    
    subparsers = parser.add_subparsers(dest='command', title='commands',
                                       description='Valid commands:',
                                       help="Run 'dolphin-logger <command> -h' for more information on a specific command.")
    subparsers.required = False # Make subcommands optional, so default behavior can be server

    # Server command (default if no command is specified)
    server_parser = subparsers.add_parser('server', help='Run the proxy server (default action if no command is given).')
    server_parser.add_argument('--port', type=int, default=None, help='Port to run the server on (default: 5001, or PORT environment variable)')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload logs to Hugging Face Hub.')
    # upload_parser.set_defaults(func=_run_upload_command)

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize dolphin-logger configuration (create default config.json).')
    init_parser.add_argument('--interactive', action='store_true', help='Interactive setup with guided API key configuration.')

    # Config command
    config_parser = subparsers.add_parser('config', help='Manage or inspect configuration.')
    config_parser.add_argument('--path', action='store_true', help='Show the expected path to the configuration file.')
    config_parser.add_argument('--validate', action='store_true', help='Validate the current configuration file.')
    config_parser.add_argument('--status', action='store_true', help='Show detailed configuration status.')
    config_parser.add_argument('--set-key', metavar='PROVIDER', help='Interactively set API key for a provider (anthropic, openai, google).')

    args = parser.parse_args()

    # --- Command Dispatching ---
    command_to_run = args.command
    if command_to_run is None: # If no command is specified, default to 'server'
        command_to_run = 'server'

    # Initialize configuration for server and upload commands.
    # For 'init' and 'config --path', it's not strictly needed beforehand,
    # but it doesn't hurt as get_config_dir() ensures the directory exists.
    # 'config --validate' relies on load_config().
    if command_to_run in ['server', 'upload']:
        try:
            print("Initializing Dolphin Logger configuration for server/upload...")
            load_config() # Ensures config dir exists, default config is copied if needed, and ENV vars are processed for server.
            print("Configuration check/setup complete for server/upload.")
        except Exception as e:
            print(f"Error during configuration initialization for {command_to_run}: {e}")
            print("Please check your setup. Exiting.")
            return

    # Dispatch based on command
    if command_to_run == 'server':
        print("Server mode activated.")
        try:
            # Handle port argument - check both main parser and subparser
            port = None
            if hasattr(args, 'port') and args.port is not None:
                port = args.port
            run_server_main(port=port) 
        except Exception as e:
            print(f"An error occurred while trying to start the server: {e}")
    
    elif command_to_run == 'upload':
        print("Upload mode activated.")
        try:
            upload_logs()
        except Exception as e:
            print(f"An error occurred during log upload: {e}")

    elif command_to_run == 'init':
        _handle_init_command()

    elif command_to_run == 'config':
        _handle_config_command(args) # Pass all args, handler will use relevant ones

    # No 'else' needed if subparsers.required = True or if default is set,
    # but with required = False and a default command logic, this is fine.

if __name__ == '__main__':
    main_cli()
