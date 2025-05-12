import os
import json
import requests
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from threading import Lock
import uuid
from datetime import date, datetime
import anthropic
from pathlib import Path
import argparse
import shutil
import glob
from huggingface_hub import HfApi, CommitOperationAdd

# Import config utility functions from the same package
from .config_utils import get_config_path, get_logs_dir, load_config

log_lock = Lock()
app = Flask(__name__)

# Enable CORS with explicit configuration
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

# Globals for daily log file management
current_logfile_name = None
current_logfile_date = None

# Global for model config, loaded when server starts
MODEL_CONFIG = []

# --- Upload Logic (from upload.py) ---
DATASET_REPO_ID = "cognitivecomputations/dolphin-logger" # Default, consider making configurable

def find_jsonl_files(log_dir):
    """Finds all .jsonl files in the specified log directory."""
    return glob.glob(os.path.join(log_dir, "*.jsonl"))

def upload_logs():
    """Uploads .jsonl files from the log directory to Hugging Face Hub and creates a PR."""
    api = HfApi()
    log_dir = get_logs_dir()
    jsonl_files = find_jsonl_files(log_dir)

    if not jsonl_files:
        print(f"No .jsonl files found in {log_dir} to upload.")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s) in {log_dir}: {jsonl_files}")

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

        commit_message = f"Add new log files from {log_dir}"

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
        print(f"An error occurred during upload: {e}")


# --- Server Logic ---

def get_current_log_file():
    global current_logfile_name, current_logfile_date
    today = date.today()
    logs_dir = get_logs_dir() # Use the dedicated logs directory

    with log_lock: # Protect access to globals
        if current_logfile_name is None: # Process just started or first call
            latest_log_file_today = None
            latest_mod_time = 0.0

            # Scan the logs directory
            for item_name in os.listdir(logs_dir):
                if item_name.endswith(".jsonl"):
                    # Check if the filename part before .jsonl is a valid UUID
                    try:
                        uuid_part = item_name[:-6] # Remove .jsonl
                        uuid.UUID(uuid_part) # Validate if it's a UUID
                    except ValueError:
                        continue # Not a UUID-named .jsonl file, skip

                    filepath = logs_dir / item_name # Use Path object for joining
                    try:
                        mod_time_timestamp = os.path.getmtime(filepath)
                        mod_date_obj = datetime.fromtimestamp(mod_time_timestamp).date()

                        if mod_date_obj == today:
                            if mod_time_timestamp > latest_mod_time:
                                latest_mod_time = mod_time_timestamp
                                latest_log_file_today = item_name
                    except OSError: # e.g., file deleted during scan, or permission issue
                        continue # Skip this file

            if latest_log_file_today:
                current_logfile_name = logs_dir / latest_log_file_today # Store full path
                current_logfile_date = today
                print(f"Resuming log file: {current_logfile_name} for date: {current_logfile_date}")
            else:
                # No log file from today found, or no valid UUID log files at all, create a new one
                new_uuid = uuid.uuid4()
                new_logfile_path = logs_dir / f"{new_uuid}.jsonl"
                current_logfile_name = new_logfile_path
                current_logfile_date = today
                print(f"Creating new log file: {current_logfile_name} for date: {current_logfile_date}")

        elif current_logfile_date != today: # Date has changed since last log (while process is running)
            new_uuid = uuid.uuid4()
            new_logfile_path = logs_dir / f"{new_uuid}.jsonl"
            current_logfile_name = new_logfile_path
            current_logfile_date = today
            print(f"Switching log file to new day: {current_logfile_name} for date: {current_logfile_date}")

    # Return the Path object directly
    return current_logfile_name

# Handle preflight OPTIONS requests explicitly
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    resp = app.make_default_options_response()
    return resp

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    if path.endswith('/models') or path.endswith('/models/'): # Handle trailing slash
        # Generate models list from MODEL_CONFIG
        models_response = []

        if MODEL_CONFIG:
            # Use models from config.json - ensure we use the "model" field, NOT the "providerModel"
            for model_config in MODEL_CONFIG:
                model_id = model_config.get("model")  # This is what clients will request
                provider = model_config.get("provider", "unknown")

                if model_id:
                    models_response.append({
                        "id": model_id,  # The model ID clients should use in requests
                        "object": "model",
                        "created": 1686935002,
                        "owned_by": provider,
                        # Include additional helpful information
                        "provider": provider,
                        "provider_model": model_config.get("providerModel", "")
                    })
                    print(f"Added model to API response: {model_id} (provider: {provider})")
        else:
            # Fallback to hardcoded models (Consider removing if config is mandatory)
            print("Warning: No models loaded from config. Using fallback models.")
            models_response = [
                {
                    "id": "gpt-fallback",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai"
                },
                {
                    "id": "claude-fallback",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "anthropic"
                },
                {
                    "id": "gemini-fallback",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "google"
                }
            ]

        resp = Response(
            json.dumps({"data": models_response, "object": "list"}), # OpenAI compatible list format
            content_type='application/json'
        )

        # Add CORS headers explicitly
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'

        return resp

    # Initialize variables
    target_api_url = None
    target_api_key = None
    target_model = None

    data = request.get_data()
    json_data = json.loads(data.decode('utf-8')) if data else None

    if request.method == 'POST' and json_data and 'model' in json_data:
        requested_model_id = json_data.get('model')
        print(f"Requested model ID: {requested_model_id}")

        model_found = False
        for model_config in MODEL_CONFIG:
            if model_config.get("model") == requested_model_id:
                # Found the model in our config
                model_found = True
                provider = model_config.get("provider")

                if provider == "ollama":
                    # For Ollama, use the model name directly
                    target_api_url = model_config.get("apiBase", "http://localhost:11434/v1") # Allow overriding default
                    target_api_key = ""  # No API key needed for local Ollama
                    target_model = model_config.get("providerModel", model_config.get("model"))
                    print(f"Using Ollama: {target_model} at {target_api_url}")
                else:
                    if provider == "anthropic":
                        # For Anthropic, use the Anthropic SDK
                        target_api_url = "anthropic_sdk"  # Special marker to use SDK instead of REST API
                        target_api_key = model_config.get("apiKey")
                        target_model = model_config.get("providerModel", requested_model_id)
                        print(f"Using Anthropic SDK: {target_model}")
                    else:
                        # For OpenAI-compatible APIs
                        target_api_url = model_config.get("apiBase")
                        target_api_key = model_config.get("apiKey")
                        target_model = model_config.get("providerModel", requested_model_id)
                        print(f"Using {provider}: {target_model} at {target_api_url}")

                # Set the provider's actual model name in the request
                if model_config.get("providerModel"):
                    json_data['model'] = model_config.get("providerModel")
                break

        if not model_found:
            # If no models configured, maybe try a default? Or just error out.
            if not MODEL_CONFIG:
                 error_msg = f"No models configured. Cannot process request for model '{requested_model_id}'."
            else:
                 error_msg = f"Model '{requested_model_id}' not found in configured models."
            print(f"Error: {error_msg}")
            resp = Response(
                json.dumps({"error": error_msg}),
                status=400,
                content_type='application/json'
            )
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        data = json.dumps(json_data).encode('utf-8')

    if not target_api_url:
        # This case might happen for non-POST requests or if logic fails above
        error_msg = "Target API endpoint could not be determined."
        print(f"Error: {error_msg}")
        if data:
            print(f"Request data: {data.decode('utf-8')}")
        resp = Response(
            json.dumps({"error": error_msg}),
            status=500,
            content_type='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    # --- Request Proxying Logic ---
    # (This part remains largely the same, but uses the determined target_api_url, target_api_key, etc.)

    # Determine the actual path for the target API
    base_path_segment = target_api_url.rstrip('/').split('/')[-1] # e.g., 'v1'
    path_parts = path.split('/', 1) # e.g. 'v1/chat/completions' -> ['v1', 'chat/completions']
    actual_path = path_parts[1] if len(path_parts) > 1 else '' # 'chat/completions'

    # Construct the final URL
    base_url_for_request = target_api_url.rstrip('/')
    if base_url_for_request.endswith(f'/{base_path_segment}'):
         base_url_for_request = base_url_for_request[:-len(f'/{base_path_segment}')]

    # Handle the special case where target_api_url is the SDK marker
    if target_api_url != "anthropic_sdk":
        url = f"{base_url_for_request}/{base_path_segment}/{actual_path}"
        print(f"Proxying request to: {url}")
    else:
        url = "anthropic_sdk" # Keep the marker for logic below
        print(f"Using Anthropic SDK for path: {actual_path}") # Log the intended path

    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'authorization', 'content-length', 'connection']} # Added 'connection'
    if target_api_url != "anthropic_sdk":
        headers['Host'] = url.split('//')[-1].split('/')[0] # Set Host for non-SDK requests
    if target_api_key: # Only add Auth header if key exists
        headers['Authorization'] = f'Bearer {target_api_key}'

    is_stream = json_data.get('stream', False) if json_data else False

    # Special handling for Anthropic SDK
    if target_api_url == "anthropic_sdk":
        print(f"DEBUG - Using Anthropic SDK for request")
        if data:
            print(f"DEBUG - Request data: {data.decode('utf-8')}")

        try:
            json_data = json.loads(data.decode('utf-8')) if data else {}

            # Initialize Anthropic client
            if not target_api_key:
                raise ValueError("Anthropic API key is missing in the configuration.")
            anthropic_client = anthropic.Anthropic(api_key=target_api_key)

            # Convert the request data to Anthropic SDK format
            messages = json_data.get('messages', [])
            max_tokens = json_data.get('max_tokens', 4096) # Default max_tokens
            system_prompt = json_data.get('system') # Handle optional system prompt

            # Prepare arguments for Anthropic client
            anthropic_args = {
                "model": target_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": is_stream,
            }
            if system_prompt:
                anthropic_args["system"] = system_prompt

            if is_stream:
                print(f"DEBUG - Creating streaming request to Anthropic API")
                stream = anthropic_client.messages.create(**anthropic_args)

                def generate():
                    response_content = ''
                    log_entry = {'request': json_data, 'response': None} # Prepare log entry
                    try:
                        for chunk in stream:
                            if chunk.type == "content_block_delta":
                                delta_content = chunk.delta.text
                                response_content += delta_content

                                # Format in OpenAI-compatible streaming format
                                openai_compatible_chunk = {
                                    "choices": [
                                        {
                                            "delta": {"content": delta_content},
                                            "index": 0,
                                            "finish_reason": None
                                        }
                                    ],
                                    "id": f"chatcmpl-anthropic-{uuid.uuid4()}", # Unique chunk ID
                                    "model": target_model, # Use the actual model name
                                    "object": "chat.completion.chunk",
                                    "created": int(datetime.now().timestamp())
                                }
                                yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                            elif chunk.type == "message_stop":
                                # Get finish reason if available (might need adjustment based on SDK)
                                finish_reason = "stop" # Default or extract from chunk if possible
                                final_chunk = {
                                     "choices": [
                                        {
                                            "delta": {}, # Empty delta for final chunk
                                            "index": 0,
                                            "finish_reason": finish_reason
                                        }
                                    ],
                                    "id": f"chatcmpl-anthropic-{uuid.uuid4()}",
                                    "model": target_model,
                                    "object": "chat.completion.chunk",
                                    "created": int(datetime.now().timestamp())
                                }
                                yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')


                        # Send final [DONE] message
                        yield b"data: [DONE]\n\n"

                    finally: # Ensure logging happens even if stream breaks
                        # Log the request/response
                        if _should_log_request(json_data):
                            log_entry['response'] = response_content # Store accumulated content
                            log_file_path = get_current_log_file()
                            with log_lock:
                                with open(log_file_path, 'a') as log_file:
                                    log_file.write(json.dumps(log_entry) + '\n')

                resp = Response(
                    stream_with_context(generate()),
                    content_type='text/event-stream'
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp
            else:
                print(f"DEBUG - Creating non-streaming request to Anthropic API")
                response_obj = anthropic_client.messages.create(**anthropic_args)

                # Convert Anthropic response to OpenAI format
                response_content = response_obj.content[0].text if response_obj.content else ""
                response_data = {
                    "id": f"chatcmpl-anthropic-{response_obj.id}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()), # Use current time
                    "model": response_obj.model, # Use model from response
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": response_obj.role, # Use role from response
                                "content": response_content
                            },
                            "finish_reason": response_obj.stop_reason
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response_obj.usage.input_tokens,
                        "completion_tokens": response_obj.usage.output_tokens,
                        "total_tokens": response_obj.usage.input_tokens + response_obj.usage.output_tokens
                    }
                }

                print(f"DEBUG - Response from Anthropic API received")
                print(f"DEBUG - Response preview: {response_content[:500]}...")

                # Log the request if needed
                if _should_log_request(json_data):
                    log_file_path = get_current_log_file()
                    with log_lock:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(json.dumps({
                                'request': json_data,
                                'response': response_content
                            }) + '\n')

                resp = Response(
                    json.dumps(response_data),
                    content_type='application/json'
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp

        except Exception as e:
            print(f"ERROR DETAILS (Anthropic SDK):")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            if data:
                print(f"  Full request data: {data.decode('utf-8')}")

            error_status = 500
            error_content_type = 'application/json'
            error_body = {"error": {"message": str(e), "type": type(e).__name__}}

            resp = Response(
                json.dumps(error_body),
                status=error_status,
                content_type=error_content_type
            )
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    # Standard REST API handling for other providers (Ollama, OpenAI-compatible)
    print(f"DEBUG - Sending request: {request.method} {url}")
    print(f"DEBUG - Headers: {headers}")
    if data:
        print(f"DEBUG - Request data: {data.decode('utf-8')}")

    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=data,
            stream=is_stream,
            timeout=300 # Add a timeout
        )

        # Print response status for debugging
        print(f"DEBUG - Response status: {response.status_code}")

        # Print a snippet of the response for debugging
        if not is_stream:
            try:
                print(f"DEBUG - Response preview: {response.text[:500]}...")
            except Exception as preview_err:
                print(f"DEBUG - Could not get response preview: {preview_err}")

        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Response Handling & Logging ---
        log_entry = {'request': json_data, 'response': None} # Prepare log entry

        if is_stream:
            def generate():
                response_content = ''
                try:
                    for line in response.iter_lines():
                        if line:
                            # Yield raw line first
                            yield line + b'\n\n' # Ensure SSE format

                            # Attempt to parse for logging
                            if line.startswith(b'data: '):
                                line_data = line.decode('utf-8')[6:]
                                if line_data != '[DONE]':
                                    try:
                                        parsed = json.loads(line_data)
                                        choices = parsed.get('choices', [])
                                        if choices and isinstance(choices, list) and len(choices) > 0:
                                            delta = choices[0].get('delta', {})
                                            if delta and isinstance(delta, dict):
                                                 delta_content = delta.get('content', '')
                                                 if delta_content and isinstance(delta_content, str):
                                                     response_content += delta_content
                                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                                        print(f"Error processing stream line for logging: {line_data}, Error: {e}")
                                        pass # Continue yielding lines even if parsing fails
                finally: # Ensure logging happens
                    if _should_log_request(json_data):
                        log_entry['response'] = response_content
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')

            resp = Response(
                stream_with_context(generate()),
                content_type=response.headers.get('Content-Type', 'text/event-stream') # Default to event-stream
            )
            # Copy relevant headers from the original response
            for hdr in ['Cache-Control', 'Content-Type']: # Add others if needed
                 if hdr in response.headers:
                     resp.headers[hdr] = response.headers[hdr]
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            # Handle non-streaming response
            complete_response = ''
            response_data = {}
            try:
                response_data = response.json()
                choices = response_data.get('choices', [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get('message', {})
                    if message and isinstance(message, dict):
                        complete_response = message.get('content', '')
                log_entry['response'] = complete_response # Log extracted content or full JSON? Let's log extracted.
            except json.JSONDecodeError:
                print("Warning: Response was not JSON. Logging raw text.")
                complete_response = response.text
                log_entry['response'] = complete_response # Log raw text

            # Log the request/response
            if _should_log_request(json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps(log_entry) + '\n')

            # Create a Flask Response object to add headers
            resp = Response(
                response.content, # Send original content back
                content_type=response.headers.get('Content-Type', 'application/json'),
                status=response.status_code
            )
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except requests.exceptions.RequestException as e:
        print(f"ERROR DETAILS (Requests):")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Method: {request.method}")
        print(f"  URL: {url if target_api_url != 'anthropic_sdk' else 'Anthropic SDK Call'}")
        # Avoid printing sensitive headers like Authorization
        safe_headers = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
        print(f"  Safe request headers: {safe_headers}")
        if data:
            # Be cautious about logging potentially large/sensitive data
            try:
                 print(f"  Request data preview: {data.decode('utf-8')[:500]}...")
            except:
                 print("  Could not decode request data preview.")


        error_content_type = 'application/json'
        error_status = 500
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}

        # Try to get more specific error info from the response, if available
        if e.response is not None:
            print(f"  Response status code: {e.response.status_code}")
            error_status = e.response.status_code
            error_content_type = e.response.headers.get('Content-Type', 'application/json')
            try:
                # Try to parse JSON error from upstream
                error_body = e.response.json()
                print(f"  Upstream error response: {json.dumps(error_body)}")
            except json.JSONDecodeError:
                # If not JSON, use the text content
                error_body = {"error": {"message": e.response.text, "type": "upstream_error"}}
                print(f"  Upstream error response (text): {e.response.text}")

        resp = Response(
            json.dumps(error_body),
            status=error_status,
            content_type=error_content_type
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e: # Catch any other unexpected errors
        print(f"UNEXPECTED SERVER ERROR:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging

        resp = Response(
            json.dumps({"error": {"message": "An unexpected internal server error occurred.", "type": type(e).__name__}}),
            status=500,
            content_type='application/json'
        )
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


def _should_log_request(current_request_data):
    """
    Determines if a request should be logged based on its content.
    Does not log if the first 'user' message content starts with '### Task'.
    """
    if not current_request_data or 'messages' not in current_request_data or not isinstance(current_request_data['messages'], list):
        return True # Log if structure is unexpected or no messages field

    for message in current_request_data['messages']:
        if isinstance(message, dict) and message.get('role') == 'user':
            content = message.get('content')
            # Check if content is a string before calling startswith
            if isinstance(content, str) and content.strip().startswith("### Task"):
                print("Skipping log for request starting with '### Task'")
                return False # First user message starts with "### Task", so DO NOT log
            # Found the first user message, and it doesn't start with the skip phrase
            return True
    return True # No user message found, or other conditions, so log by default

def run_server():
    """Sets up and runs the Flask development server."""
    global MODEL_CONFIG
    # Load config when server starts
    try:
        MODEL_CONFIG = load_config().get('models', [])
        print(f"Loaded {len(MODEL_CONFIG)} models from config")
    except Exception as e:
        print(f"Error loading config file: {e}")
        MODEL_CONFIG = []

    # Note about potential port differences
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Dolphin Logger server on port {port}...")
    print(f"Configuration loaded from: {get_config_path()}")
    print(f"Logs will be stored in: {get_logs_dir()}")
    print(f"Loaded {len(MODEL_CONFIG)} model configurations.")
    # Use waitress or gunicorn for production instead of app.run(debug=True)
    # For simplicity in development/local use, app.run is okay.
    # Consider adding host='0.0.0.0' if you need to access it from other devices on your network
    app.run(host='0.0.0.0', port=port, debug=False) # Turn off debug mode for installs

def cli():
    """Command Line Interface entry point."""
    parser = argparse.ArgumentParser(description="Dolphin Logger: Proxy server and log uploader.")
    parser.add_argument('--upload', action='store_true', help='Upload logs to Hugging Face Hub instead of starting the server.')

    args = parser.parse_args()

    # Ensure config directory and default config exist before running either mode
    try:
        load_config() # This will create the dir and copy default config if needed
    except Exception as e:
        print(f"Error ensuring configuration exists: {e}")
        # Decide if we should exit or continue depending on the error

    if args.upload:
        print("Upload mode activated.")
        upload_logs()
    else:
        print("Server mode activated.")
        run_server()

# This ensures the cli function is called when the script is run directly
# (though the primary entry point is via pyproject.toml)
if __name__ == "__main__":
    cli()
