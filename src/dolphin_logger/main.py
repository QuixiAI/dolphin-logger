import os
import json
import requests
from flask import Flask, request, Response, stream_with_context, jsonify
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


def _get_target_api_config(requested_model_id, json_data, model_config_list):
    """
    Determines the target API URL, key, and model based on the requested model ID
    and the server's model configuration.

    Args:
        requested_model_id (str): The model ID from the client's request.
        json_data (dict): The parsed JSON data from the request.
        model_config_list (list): The MODEL_CONFIG list.

    Returns:
        dict: A dictionary containing 'target_api_url', 'target_api_key',
              'target_model', 'provider', and 'error' (if any).
    """
    for model_config in model_config_list:
        if model_config.get("model") == requested_model_id:
            provider = model_config.get("provider")
            target_model = model_config.get("providerModel", requested_model_id)
            api_key = model_config.get("apiKey")
            api_base = model_config.get("apiBase")

            if provider == "ollama":
                return {
                    "target_api_url": api_base or "http://localhost:11434/v1",
                    "target_api_key": "",  # No API key for Ollama
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }
            elif provider == "anthropic":
                return {
                    "target_api_url": "anthropic_sdk", # Special marker
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }
            else: # OpenAI-compatible
                if not api_base:
                    return {"error": f"apiBase not configured for model '{requested_model_id}'"}
                return {
                    "target_api_url": api_base,
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }

    if not model_config_list:
        return {"error": f"No models configured. Cannot process request for model '{requested_model_id}'."}
    return {"error": f"Model '{requested_model_id}' not found in configured models."}


# Handle preflight OPTIONS requests explicitly
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    resp = app.make_default_options_response()
    return resp

@app.route('/health', methods=['GET'])
def health_check():
    """Provides a health check endpoint for the server."""
    if MODEL_CONFIG: # Check if MODEL_CONFIG is loaded and not empty
        response_data = {"status": "ok", "message": "Server is healthy, configuration loaded."}
        status_code = 200
    else:
        # This case implies that load_config() might have returned an empty 'models' list
        # or MODEL_CONFIG was not populated correctly.
        response_data = {"status": "error", "message": "Server is running, but configuration might have issues (e.g., no models loaded)."}
        status_code = 500
    
    resp = jsonify(response_data)
    resp.status_code = status_code
    # jsonify already sets Content-Type to application/json
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

    data = request.get_data()
    # We need json_data early for model selection and logging decisions
    try:
        # Decode once, use everywhere
        decoded_data = data.decode('utf-8') if data else "{}" # Handle empty data
        json_data = json.loads(decoded_data) if decoded_data else {}
    except json.JSONDecodeError:
        err_resp = Response(json.dumps({"error": "Invalid JSON in request body"}), status=400, content_type='application/json')
        err_resp.headers['Access-Control-Allow-Origin'] = '*'
        return err_resp

    if request.method != 'POST' or 'model' not in json_data:
        error_msg = "Proxying requires a POST request with a 'model' field in JSON body."
        if request.method == 'POST' and 'model' not in json_data :
            error_msg = "POST request JSON body must include a 'model' field."

        print(f"Error: {error_msg}")
        err_resp = Response(
            json.dumps({"error": error_msg}),
            status=400, # Bad Request
            content_type='application/json'
        )
        err_resp.headers['Access-Control-Allow-Origin'] = '*'
        return err_resp

    requested_model_id = json_data.get('model')
    print(f"Requested model ID: {requested_model_id}")

    api_config = _get_target_api_config(requested_model_id, json_data, MODEL_CONFIG)

    if api_config.get("error"):
        error_msg = api_config["error"]
        print(f"Error getting API config: {error_msg}")
        err_resp = Response(
            json.dumps({"error": error_msg}),
            status=400, # Bad Request, as it's a config/request matching issue
            content_type='application/json'
        )
        err_resp.headers['Access-Control-Allow-Origin'] = '*'
        return err_resp

    target_api_url = api_config["target_api_url"]
    target_api_key = api_config["target_api_key"]
    target_model = api_config["target_model"] # This is the providerModel
    # provider = api_config["provider"] # Available if needed for future logic

    # This will be the dict passed to Anthropic SDK or used to create JSON for REST
    json_data_for_downstream = json_data.copy()
    if target_model != requested_model_id: # If providerModel is different from what client sent
        json_data_for_downstream['model'] = target_model

    # data_to_send_bytes will be used for REST POST/PUT bodies
    data_to_send_bytes = json.dumps(json_data_for_downstream).encode('utf-8') if json_data_for_downstream else data

    is_stream = json_data.get('stream', False)

    # --- Delegate to provider-specific handlers ---
    # original_request_json_data is json_data (what the client originally sent)
    # json_data_for_downstream is what we send to the target (potentially modified model field)
    # data_to_send_bytes is the encoded version of json_data_for_downstream for REST

    if target_api_url == "anthropic_sdk":
        return _handle_anthropic_sdk_request(
            json_data_for_sdk=json_data_for_downstream, # This has the correct model for Anthropic
            target_model=target_model, # This is the providerModel from config
            target_api_key=target_api_key,
            is_stream=is_stream,
            original_request_json_data=json_data # For logging the original client request
        )
    else:
        # Construct URL for REST APIs (Ollama, OpenAI-compatible)
        # The `path` variable from Flask route is used here.
        # e.g., if incoming request is to /v1/chat/completions, path is 'v1/chat/completions'
        # target_api_url is the base URL of the target API, e.g., http://localhost:11434/v1
        # The final URL should be target_api_url + actual path from the request
        # Ensure no double slashes if target_api_url ends with / and path starts with /
        url = f"{target_api_url.rstrip('/')}/{path.lstrip('/')}"
        print(f"Proxying REST request to: {url}")

        headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'authorization', 'content-length', 'connection', 'user-agent']}
        headers['Host'] = target_api_url.split('//')[-1].split('/')[0] # Set Host to target API's host
        if target_api_key:
            headers['Authorization'] = f'Bearer {target_api_key}'
        # Preserve user-agent if you want the target to see it, or set your own.
        # headers['User-Agent'] = request.headers.get('User-Agent', 'DolphinLoggerProxy/1.0')


        return _handle_rest_api_request(
            method=request.method,
            url=url,
            headers=headers,
            data_bytes=data_to_send_bytes, # Use encoded bytes with potentially modified model
            is_stream=is_stream,
            original_request_json_data=json_data # For logging original client request
        )


def _handle_anthropic_sdk_request(json_data_for_sdk, target_model, target_api_key, is_stream, original_request_json_data):
    """
    Handles requests to the Anthropic SDK.
    `json_data_for_sdk` is the dictionary to be used for constructing the Anthropic SDK call.
    `original_request_json_data` is the dictionary from the initial request, used for logging.
    """
    print(f"DEBUG - Using Anthropic SDK for request with model: {target_model}")
    # If data is sensitive, avoid logging the full body here or use a preview.
    # print(f"DEBUG - Request data for SDK: {json.dumps(json_data_for_sdk)}")

    try:
        if not target_api_key:
            # This is a server configuration error, should ideally be caught at startup
            # or result in a clear error message if a model needing a key is chosen without one.
            raise ValueError("Anthropic API key is missing in the configuration for the requested model.")
        
        anthropic_client = anthropic.Anthropic(api_key=target_api_key)

        messages = json_data_for_sdk.get('messages', [])
        max_tokens = json_data_for_sdk.get('max_tokens', 4096) # Anthropic's default
        system_prompt = json_data_for_sdk.get('system') # Optional system prompt

        anthropic_args = {
            "model": target_model, # This is the providerModel from config
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": is_stream,
        }
        if system_prompt: # Add system prompt only if provided
            anthropic_args["system"] = system_prompt

        if is_stream:
            print(f"DEBUG - Creating streaming request to Anthropic API: model={target_model}")
            sdk_stream = anthropic_client.messages.create(**anthropic_args)

            def generate_anthropic_stream_response():
                response_content_parts = []
                log_entry = {'request': original_request_json_data, 'response': None}
                try:
                    for chunk in sdk_stream:
                        if chunk.type == "content_block_delta":
                            delta_content = chunk.delta.text
                            response_content_parts.append(delta_content)
                            openai_compatible_chunk = {
                                "choices": [{"delta": {"content": delta_content}, "index": 0, "finish_reason": None}],
                                "id": f"chatcmpl-anthropic-{uuid.uuid4()}",
                                "model": target_model, 
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                        elif chunk.type == "message_stop":
                            # Anthropic SDK provides stop_reason in the message object upon stop
                            finish_reason = chunk.message.stop_reason if hasattr(chunk, 'message') and hasattr(chunk.message, 'stop_reason') else "stop"
                            final_chunk = {
                                 "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                                "id": f"chatcmpl-anthropic-{uuid.uuid4()}",
                                "model": target_model,
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                finally:
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')

            resp = Response(stream_with_context(generate_anthropic_stream_response()), content_type='text/event-stream')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else: # Non-streaming Anthropic request
            print(f"DEBUG - Creating non-streaming request to Anthropic API: model={target_model}")
            response_obj = anthropic_client.messages.create(**anthropic_args)
            
            response_content = response_obj.content[0].text if response_obj.content and len(response_obj.content) > 0 and hasattr(response_obj.content[0], 'text') else ""
            
            # Convert Anthropic response to OpenAI-compatible format
            response_data_converted = {
                "id": f"chatcmpl-anthropic-{response_obj.id}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()), # Consider using a more accurate timestamp if available
                "model": response_obj.model, # Use model from Anthropic's response
                "choices": [{
                    "index": 0,
                    "message": {"role": response_obj.role, "content": response_content},
                    "finish_reason": response_obj.stop_reason 
                }],
                "usage": { # Anthropic provides usage data
                    "prompt_tokens": response_obj.usage.input_tokens,
                    "completion_tokens": response_obj.usage.output_tokens,
                    "total_tokens": response_obj.usage.input_tokens + response_obj.usage.output_tokens
                }
            }
            print(f"DEBUG - Response from Anthropic API received. Preview: {response_content[:100]}...")
            if _should_log_request(original_request_json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    log_file.write(json.dumps({'request': original_request_json_data, 'response': response_content}) + '\n')
            
            resp = Response(json.dumps(response_data_converted), content_type='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except anthropic.APIError as e: # Catch specific Anthropic errors
        print(f"ERROR DETAILS (Anthropic SDK - APIError): Status: {e.status_code}, Type: {e.type}, Message: {e.message}")
        error_body = {"error": {"message": e.message, "type": e.type or type(e).__name__, "code": e.status_code}}
        resp = Response(json.dumps(error_body), status=e.status_code or 500, content_type='application/json')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e: # Catch other errors like ValueError from misconfiguration
        print(f"ERROR DETAILS (Anthropic SDK - General): Type: {type(e).__name__}, Message: {str(e)}")
        if original_request_json_data:
            print(f"  Full request data (original for logging): {json.dumps(original_request_json_data)}")
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}
        resp = Response(json.dumps(error_body), status=500, content_type='application/json')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


def _handle_rest_api_request(method, url, headers, data_bytes, is_stream, original_request_json_data):
    """
    Handles requests to OpenAI-compatible (including Ollama) REST APIs.
    `data_bytes` is the already encoded byte string for the request body.
    `original_request_json_data` is the parsed JSON from the original client request, for logging.
    """
    print(f"DEBUG - Sending REST request: {method} {url}")
    # print(f"DEBUG - Headers: {headers}") # Potentially sensitive (e.g. API keys in some non-Bearer setups)
    # if data_bytes:
    #     try:
    #         print(f"DEBUG - Request data preview: {data_bytes.decode('utf-8')[:200]}...")
    #     except UnicodeDecodeError:
    #         print(f"DEBUG - Request data: <binary or non-utf8 data>")

    try:
        api_response = requests.request(
            method=method,
            url=url,
            headers=headers,
            data=data_bytes,
            stream=is_stream,
            timeout=300 # Standard timeout
        )
        print(f"DEBUG - REST API Response status: {api_response.status_code}")
        # if not is_stream:
        #     try:
        #         print(f"DEBUG - REST API Response preview: {api_response.text[:200]}...")
        #     except Exception as preview_err:
        #         print(f"DEBUG - Could not get REST API response preview: {preview_err}")

        api_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        log_entry = {'request': original_request_json_data, 'response': None}

        if is_stream:
            def generate_rest_stream_response():
                response_content_parts = []
                try:
                    for line in api_response.iter_lines():
                        if line:
                            yield line + b'\n\n' # Pass through to client, maintaining SSE
                            if line.startswith(b'data: '):
                                line_data_str = line.decode('utf-8', errors='replace')[6:]
                                if line_data_str.strip() != '[DONE]':
                                    try:
                                        parsed_chunk = json.loads(line_data_str)
                                        choices = parsed_chunk.get('choices', [])
                                        if choices and isinstance(choices, list) and len(choices) > 0:
                                            delta = choices[0].get('delta', {})
                                            if delta and isinstance(delta, dict):
                                                delta_content = delta.get('content', '')
                                                if delta_content and isinstance(delta_content, str):
                                                    response_content_parts.append(delta_content)
                                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                                        # Error parsing a stream chunk for logging, but stream to client continues
                                        # print(f"Warning: Error processing stream line for REST logging: {line_data_str}, Error: {e_parse}")
                                        pass # Don't let logging error break the stream to client
                finally:
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')
            
            resp = Response(stream_with_context(generate_rest_stream_response()), content_type=api_response.headers.get('Content-Type', 'text/event-stream'))
            for hdr_key in ['Cache-Control', 'Content-Type', 'Transfer-Encoding', 'Date', 'Server']: # Common headers to propagate
                if hdr_key in api_response.headers:
                    resp.headers[hdr_key] = api_response.headers[hdr_key]
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else: # Non-streaming REST response
            complete_response_text = ''
            try:
                # Try to parse as JSON, as this is typical for chat completion APIs
                response_json = api_response.json()
                # Standard OpenAI format for extracting content
                choices = response_json.get('choices', [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get('message', {})
                    if message and isinstance(message, dict):
                        complete_response_text = message.get('content', '')
                # If not OpenAI format, or content not found, log the whole JSON
                if not complete_response_text:
                    complete_response_text = response_json # Or json.dumps(response_json) for string
                log_entry['response'] = complete_response_text
            except json.JSONDecodeError:
                # If response is not JSON, log raw text
                print("Warning: REST API Response was not JSON. Logging raw text.")
                complete_response_text = api_response.text
                log_entry['response'] = complete_response_text

            if _should_log_request(original_request_json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps(log_entry) + '\n')
            
            # Return the raw response content from the target API to the client
            resp = Response(api_response.content, content_type=api_response.headers.get('Content-Type', 'application/json'), status=api_response.status_code)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        print(f"ERROR DETAILS (REST API - RequestException): Type: {type(e).__name__}, Message: {str(e)}, URL: {url}")
        # safe_headers_for_err = {k: v for k,v in headers.items() if k.lower() != 'authorization'} # Avoid logging auth
        # print(f"  Safe request headers: {safe_headers_for_err}")
        # if data_bytes:
        #     try:
        #         print(f"  Request data preview: {data_bytes.decode('utf-8')[:200]}...")
        #     except:
        #         print("  Could not decode request data preview for error log.")

        error_content_type = 'application/json'
        error_status = 502 # Bad Gateway, as we failed to get a response from upstream
        error_body_msg = f"Error connecting to upstream API: {str(e)}"
        error_body = {"error": {"message": error_body_msg, "type": type(e).__name__}}

        if e.response is not None:
            # If the error is an HTTPError (e.g., 4xx, 5xx from upstream), use its details
            print(f"  Upstream response status code: {e.response.status_code}")
            error_status = e.response.status_code # Use upstream's status code
            error_content_type = e.response.headers.get('Content-Type', 'application/json')
            try:
                error_body = e.response.json() # Try to parse upstream error
                print(f"  Upstream error response JSON: {json.dumps(error_body)}")
            except json.JSONDecodeError:
                error_body = {"error": {"message": e.response.text, "type": "upstream_error"}}
                print(f"  Upstream error response Text: {e.response.text}")
        
        resp = Response(json.dumps(error_body), status=error_status, content_type=error_content_type)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e: # Catch-all for any other unexpected errors in this handler
        print(f"UNEXPECTED REST API HANDLER ERROR: Type: {type(e).__name__}, Message: {str(e)}")
        import traceback
        traceback.print_exc()
        resp = Response(json.dumps({"error": {"message": "An unexpected error occurred in the REST API handler.", "type": type(e).__name__}}), status=500, content_type='application/json')
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
