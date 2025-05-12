import os
import json
import requests
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS  # Import Flask-CORS for proper cross-origin handling
from threading import Lock
import uuid
from datetime import date, datetime
import anthropic  # Import Anthropic SDK

log_lock = Lock()
app = Flask(__name__)

# Enable CORS with explicit configuration
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

# Globals for daily log file management
current_logfile_name = None
current_logfile_date = None

# Load models from config.json
MODEL_CONFIG = []
try:
    with open('config.json', 'r') as f:
        model_data = json.load(f)
        MODEL_CONFIG = model_data.get('models', [])
    print(f"Loaded {len(MODEL_CONFIG)} models from config.json")
except Exception as e:
    print(f"Error loading config.json: {e}")
    MODEL_CONFIG = []

def get_current_log_file():
    global current_logfile_name, current_logfile_date
    today = date.today()

    with log_lock: # Protect access to globals
        if current_logfile_name is None: # Process just started or first call
            latest_log_file_today = None
            latest_mod_time = 0.0

            for item_name in os.listdir('.'): # Scan current directory
                if item_name.endswith(".jsonl"):
                    # Check if the filename part before .jsonl is a valid UUID
                    try:
                        uuid_part = item_name[:-6] # Remove .jsonl
                        uuid.UUID(uuid_part) # Validate if it's a UUID
                    except ValueError:
                        continue # Not a UUID-named .jsonl file, skip

                    filepath = os.path.join('.', item_name)
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
                current_logfile_name = latest_log_file_today
                current_logfile_date = today
                print(f"Resuming log file: {current_logfile_name} for date: {current_logfile_date}")
            else:
                # No log file from today found, or no valid UUID log files at all, create a new one
                new_uuid = uuid.uuid4()
                current_logfile_name = f"{new_uuid}.jsonl"
                current_logfile_date = today
                print(f"Creating new log file: {current_logfile_name} for date: {current_logfile_date}")

        elif current_logfile_date != today: # Date has changed since last log (while process is running)
            new_uuid = uuid.uuid4()
            current_logfile_name = f"{new_uuid}.jsonl"
            current_logfile_date = today
            print(f"Switching log file to new day: {current_logfile_name} for date: {current_logfile_date}")
            
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
            # Fallback to hardcoded models
            models_response = [
                {
                    "id": "gpt",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai"
                },
                {
                    "id": "claude",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "anthropic"
                },
                {
                    "id": "gemini",
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
                    target_api_url = "http://localhost:11434/v1"
                    target_api_key = ""  # No API key needed for local Ollama
                    target_model = model_config.get("providerModel", model_config.get("model"))
                    print(f"Using Ollama: {target_model}")
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
            error_msg = f"Model '{requested_model_id}' not found in config.json"
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
        error_msg = "Target API endpoint is not configured for the selected model."
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

    base_path_segment = target_api_url.rstrip('/').split('/')[-1] # e.g., 'v1'
    
    # Strip out the first path component (which is the API version like 'v1') 
    # and replace with the base_path_segment from the target_api_url
    path_parts = path.split('/', 1) # e.g. 'v1/chat/completions' -> ['v1', 'chat/completions']
    actual_path = path_parts[1] if len(path_parts) > 1 else '' # 'chat/completions'
    
    # Construct the final URL
    # Remove the base_path_segment from target_api_url if it's already there to avoid duplication
    base_url_for_request = target_api_url.rstrip('/')
    if base_url_for_request.endswith(f'/{base_path_segment}'):
        base_url_for_request = base_url_for_request[:-len(f'/{base_path_segment}')]
    
    url = f"{base_url_for_request}/{base_path_segment}/{actual_path}"
    print(f"Proxying request to: {url}")
    
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['host', 'authorization', 'content-length']}
    headers['Host'] = url.split('//')[-1].split('/')[0]
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
            anthropic_client = anthropic.Anthropic(api_key=target_api_key)
            
            # Convert the request data to Anthropic SDK format
            if is_stream:
                print(f"DEBUG - Creating streaming request to Anthropic API")
                
                # Extract messages from the request
                messages = json_data.get('messages', [])
                max_tokens = json_data.get('max_tokens', 4096)
                
                # Create streaming response using Anthropic SDK
                stream = anthropic_client.messages.create(
                    model=target_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=True,
                )
                
                def generate():
                    response_content = ''
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
                                "id": "chatcmpl-anthropic",
                                "model": target_model,
                                "object": "chat.completion.chunk"
                            }
                            
                            yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                    
                    # Send final [DONE] message
                    yield b"data: [DONE]\n\n"
                    
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
                    stream_with_context(generate()),
                    content_type='text/event-stream'
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp
            else:
                print(f"DEBUG - Creating non-streaming request to Anthropic API")
                
                # Extract messages from the request
                messages = json_data.get('messages', [])
                max_tokens = json_data.get('max_tokens', 4096)
                
                # Create non-streaming response using Anthropic SDK
                response_obj = anthropic_client.messages.create(
                    model=target_model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                
                # Convert Anthropic response to OpenAI format
                response_data = {
                    "id": "chatcmpl-anthropic",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_obj.content[0].text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": response_obj.usage.input_tokens,
                        "completion_tokens": response_obj.usage.output_tokens,
                        "total_tokens": response_obj.usage.input_tokens + response_obj.usage.output_tokens
                    }
                }
                
                print(f"DEBUG - Response from Anthropic API received")
                print(f"DEBUG - Response preview: {response_obj.content[0].text[:500]}...")
                
                # Log the request if needed
                if _should_log_request(json_data):
                    log_file_path = get_current_log_file()
                    with log_lock:
                        with open(log_file_path, 'a') as log_file:
                            log_file.write(json.dumps({
                                'request': json_data,
                                'response': response_obj.content[0].text
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
    
    # Standard REST API handling for other providers
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
        )
        
        # Print response status for debugging
        print(f"DEBUG - Response status: {response.status_code}")
        
        # Print a snippet of the response for debugging
        if not is_stream:
            try:
                print(f"DEBUG - Response preview: {response.text[:500]}...")
            except:
                print("DEBUG - Could not get response preview")
        
        response.raise_for_status()

        if request.method == 'POST':
            if is_stream:
                def generate():
                    response_content = ''
                    for line in response.iter_lines():
                        if line:
                            if line.startswith(b'data: '):
                                yield line + b'\n\n'  # Added extra newline for SSE
                                line_data = line.decode('utf-8')[6:]
                                if line_data != '[DONE]':
                                    try: # Add try-except for robustness
                                        parsed = json.loads(line_data)
                                        choices = parsed.get('choices', [])
                                        if choices and 'delta' in choices[0]:
                                            delta_content = choices[0]['delta'].get('content', '')
                                            response_content += delta_content
                                        else:
                                            print(f"Warning: Empty or malformed 'choices' in stream line: {line_data}")
                                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                                        print(f"Error processing stream line: {line_data}, Error: {e}")
                                        pass 

                    if _should_log_request(json_data):
                        log_file_path = get_current_log_file()
                        with log_lock: # Ensures atomic write to the file
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps({
                                    'request': json_data,
                                    'response': response_content
                                }) + '\n')
                
                resp = Response(
                    stream_with_context(generate()), 
                    content_type=response.headers.get('Content-Type')
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp
            else:
                response_data = response.json()
                choices = response_data.get('choices', [])
                if choices and 'message' in choices[0]:
                    complete_response = choices[0]['message'].get('content', '')
                else:
                    print(f"Warning: Empty or malformed 'choices' in non-stream response: {response_data}")
                    complete_response = ''

                if _should_log_request(json_data):
                    log_file_path = get_current_log_file()
                    with log_lock: # Ensures atomic write to the file
                        with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps({
                                    'request': json_data,
                                    'response': complete_response
                                }) + '\n')
                
                # Create a Flask Response object to add headers
                resp = Response(
                    json.dumps(response_data), 
                    content_type='application/json', 
                    status=response.status_code
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp
        else: # For GET, PUT, DELETE etc. that are not POST
            if is_stream:
                # Similar to POST streaming
                resp = Response(
                    stream_with_context(response.iter_content(chunk_size=None)), 
                    content_type=response.headers.get('Content-Type')
                )
                resp.headers['Access-Control-Allow-Origin'] = '*'
                return resp
            else:
                # For non-streaming GET, etc.
                try:
                    json_response_data = response.json()
                    resp = Response(
                        json.dumps(json_response_data), 
                        content_type='application/json', 
                        status=response.status_code
                    )
                    resp.headers['Access-Control-Allow-Origin'] = '*'
                    return resp
                except json.JSONDecodeError:
                    # If response is not JSON, return raw content
                    resp = Response(
                        response.content, 
                        content_type=response.headers.get('Content-Type'), 
                        status=response.status_code
                    )
                    resp.headers['Access-Control-Allow-Origin'] = '*'
                    return resp


    except Exception as e:
        print(f"ERROR DETAILS:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Method: {request.method}")
        print(f"  URL: {url}")
        print(f"  Full request headers: {headers}")
        if data:
            print(f"  Full request data: {data.decode('utf-8')}")
        
        # Check if it's a requests exception with a response
        if isinstance(e, requests.exceptions.RequestException) and hasattr(e, 'response') and e.response:
            print(f"  Response status code: {e.response.status_code}")
            try:
                print(f"  Response headers: {e.response.headers}")
                print(f"  Response content: {e.response.text}")
            except Exception as inner_e:
                print(f"  Could not get response details: {str(inner_e)}")
        error_content_type = 'application/json'
        error_status = 500
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}

        if hasattr(e.response, 'status_code'):
            error_status = e.response.status_code
        if hasattr(e.response, 'headers'):
            error_content_type = e.response.headers.get('Content-Type', 'application/json')
        
        if hasattr(e.response, 'json'):
            try:
                error_body = e.response.json()
            except json.JSONDecodeError:
                # If original error response is not JSON, use the generic one
                pass # error_body already set
        elif hasattr(e.response, 'text'):
             error_body = {"error": {"message": e.response.text, "type": type(e).__name__}}

        resp = Response(
            json.dumps(error_body), 
            status=error_status, 
            content_type=error_content_type
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
            if isinstance(content, str) and content.startswith("### Task"):
                print("Skipping log for request starting with '### Task'")
                return False # First user message starts with "### Task", so DO NOT log
            return True # First user message found, and it does NOT start with "### Task", so log
    return True # No user message found, or other conditions, so log by default

if __name__ == '__main__':
    # Note about potential port differences
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on port {port}. If you need to run on port 5000 instead, set the PORT environment variable.")
    
    # Use specific CORS settings to debug issues
    app.run(host='0.0.0.0', port=port, debug=True)
