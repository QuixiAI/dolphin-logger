import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, Response, stream_with_context
from threading import Lock
import uuid
from datetime import date, datetime  # Modified: added datetime

log_lock = Lock()
app = Flask(__name__)
# LOG_FILE = 'logs.jsonl' # Removed

# Globals for daily log file management
current_logfile_name = None
current_logfile_date = None

load_dotenv()
OPENAI_API_URL = os.getenv('OPENAI_ENDPOINT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

ANTHROPIC_API_URL = os.getenv('ANTHROPIC_ENDPOINT')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL')

GOOGLE_API_URL = os.getenv('GOOGLE_ENDPOINT')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_MODEL = os.getenv('GOOGLE_MODEL')

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

MODELS = [
    {
        "id": "gpt",
        "object": "model",
        "created": 1686935002,
        "owned_by": "organization-owner"
    },
    {
        "id": "claude",
        "object": "model",
        "created": 1686935002,
        "owned_by": "organization-owner"
    },
    {
        "id": "gemini",
        "object": "model",
        "created": 1686935002,
        "owned_by": "organization-owner"
    }
]

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
def proxy(path):
    if request.method == 'OPTIONS':
        # Handle CORS preflight requests
        response_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With', # Common headers
            'Access-Control-Max-Age': '86400'  # Cache preflight response for 1 day
        }
        return Response(status=200, headers=response_headers)

    if path.endswith('/models') or path.endswith('/models/'): # Handle trailing slash
        # Ensure OPENAI_MODEL is not None before using it as a default
        # and that other models are also available if their env vars are set
        
        # For the /v1/models endpoint, we return a list of available models
        # based on the MODELS constant defined above.
        # We can also dynamically check if the corresponding API keys are set
        # to only list models that are configured.
        
        available_models = []
        if OPENAI_API_KEY and OPENAI_MODEL:
             available_models.append({
                "id": OPENAI_MODEL, # Use the actual model name from env
                "object": "model",
                "created": 1686935002,
                "owned_by": "openai"
            })
        # The user specifically asked for "gpt", "claude", "gemini" to be listed.
        # The previous logic was trying to be too dynamic.
        # Let's stick to the user's request.
        
        models_response = [
            {
                "id": "gpt", # Per user request
                "object": "model",
                "created": 1686935002,
                "owned_by": "openai" # Placeholder, can be adjusted
            },
            {
                "id": "claude", # Per user request
                "object": "model",
                "created": 1686935002,
                "owned_by": "anthropic" # Placeholder
            },
            {
                "id": "gemini", # Per user request
                "object": "model",
                "created": 1686935002,
                "owned_by": "google" # Placeholder
            }
        ]
        response = Response(
            json.dumps({"data": models_response, "object": "list"}), # OpenAI compatible list format
            content_type='application/json'
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    # print("\n=== Incoming Request ===")
    # print(f"Method: {request.method}")
    # print(f"Path: {path}")
    # print(f"Headers: {dict(request.headers)}")
    # print(f"Raw Data: {request.get_data().decode('utf-8')}")

    target_api_url = OPENAI_API_URL
    target_api_key = OPENAI_API_KEY
    target_model = OPENAI_MODEL

    data = request.get_data()
    json_data = json.loads(data.decode('utf-8')) if data else None

    if request.method == 'POST' and json_data and 'model' in json_data:
        requested_model_id = json_data.get('model')
        print(f"Requested model ID: {requested_model_id}")

        if requested_model_id == "gpt" or (OPENAI_MODEL and requested_model_id == OPENAI_MODEL):
            target_api_url = OPENAI_API_URL
            target_api_key = OPENAI_API_KEY
            target_model = OPENAI_MODEL # Use the specific model name from env
            json_data['model'] = target_model # Ensure the outgoing request uses the correct model name
            print(f"Using OpenAI: {target_model} at {target_api_url}")
        elif requested_model_id == "claude" or (ANTHROPIC_MODEL and requested_model_id == ANTHROPIC_MODEL):
            target_api_url = ANTHROPIC_API_URL
            target_api_key = ANTHROPIC_API_KEY
            target_model = ANTHROPIC_MODEL
            json_data['model'] = target_model # Ensure the outgoing request uses the correct model name
            print(f"Using Anthropic: {target_model} at {target_api_url}")
        elif requested_model_id == "gemini" or (GOOGLE_MODEL and requested_model_id == GOOGLE_MODEL):
            target_api_url = GOOGLE_API_URL
            target_api_key = GOOGLE_API_KEY
            target_model = GOOGLE_MODEL
            json_data['model'] = target_model # Ensure the outgoing request uses the correct model name
            print(f"Using Google: {target_model} at {target_api_url}")
        else:
            # Default or error if model not supported or configured
            # For now, let's default to OpenAI if specific model not found or misconfigured
            # Or, return an error:
            # return Response(json.dumps({"error": f"Model {requested_model_id} not supported or configured"}), status=400, content_type='application/json')
            print(f"Warning: Model '{requested_model_id}' not explicitly handled or configured, defaulting to OpenAI.")
            target_api_url = OPENAI_API_URL
            target_api_key = OPENAI_API_KEY
            target_model = OPENAI_MODEL
            json_data['model'] = target_model


        data = json.dumps(json_data).encode('utf-8')


    if not target_api_url or not target_api_key:
        error_response = Response(json.dumps({"error": "Target API endpoint or key is not configured for the selected model."}), status=500, content_type='application/json')
        error_response.headers['Access-Control-Allow-Origin'] = '*'
        return error_response

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
    
    # print("\n=== Outgoing Request ===")
    # print(f"URL: {url}")
    # print(f"Headers: {headers}")
    # print(f"Data: {data.decode('utf-8') if data else None}")

    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=data,
            stream=is_stream,
        )
        
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
                                        delta_content = json.loads(line_data)['choices'][0]['delta'].get('content', '')
                                        response_content += delta_content
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
                
                # For streaming responses, Flask-CORS or @after_request would be better,
                # but for now, let's ensure the initial response has the header.
                # The browser checks CORS on the initial headers of the stream.
                stream_response = Response(stream_with_context(generate()), content_type=response.headers.get('Content-Type'))
                stream_response.headers['Access-Control-Allow-Origin'] = '*'
                return stream_response
            else:
                response_data = response.json()
                complete_response = response_data['choices'][0]['message']['content']

                if _should_log_request(json_data):
                    log_file_path = get_current_log_file()
                    with log_lock: # Ensures atomic write to the file
                        with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps({
                                    'request': json_data,
                                    'response': complete_response
                                }) + '\n')
                
                # Create a Flask Response object to add headers
                final_response = Response(json.dumps(response_data), content_type='application/json', status=response.status_code)
                final_response.headers['Access-Control-Allow-Origin'] = '*'
                return final_response
        else: # For GET, PUT, DELETE etc. that are not POST
            if is_stream:
                # Similar to POST streaming
                stream_response = Response(stream_with_context(response.iter_content(chunk_size=None)), content_type=response.headers.get('Content-Type'))
                stream_response.headers['Access-Control-Allow-Origin'] = '*'
                return stream_response
            else:
                # For non-streaming GET, etc.
                try:
                    json_response_data = response.json()
                    final_response = Response(json.dumps(json_response_data), content_type='application/json', status=response.status_code)
                    final_response.headers['Access-Control-Allow-Origin'] = '*'
                    return final_response
                except json.JSONDecodeError:
                    # If response is not JSON, return raw content
                    final_response = Response(response.content, content_type=response.headers.get('Content-Type'), status=response.status_code)
                    final_response.headers['Access-Control-Allow-Origin'] = '*'
                    return final_response


    except requests.exceptions.RequestException as e:
        print(f"Error proxying request: {e}")
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


        final_error_response = Response(json.dumps(error_body), 
                              status=error_status, 
                              content_type=error_content_type) # Use original content type if available
        final_error_response.headers['Access-Control-Allow-Origin'] = '*'
        return final_error_response

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
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
