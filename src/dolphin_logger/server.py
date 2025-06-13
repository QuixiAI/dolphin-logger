import os
import json
from flask import Flask, request, Response, jsonify 
from flask_cors import CORS

# Imports from our refactored modules
from .config import load_config, get_config_path, get_logs_dir
from .core_proxy import get_target_api_config, handle_anthropic_sdk_request, handle_rest_api_request
# logging_utils are used by core_proxy, not directly by server.py usually.

app = Flask(__name__)

# Enable CORS with explicit configuration
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

# Global for model config, loaded when server starts
MODEL_CONFIG: list = []


# Handle preflight OPTIONS requests explicitly
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path: str | None = None): # path can be None if accessing '/'
    resp = app.make_default_options_response()
    return resp

@app.route('/health', methods=['GET'])
def health_check():
    """Provides a health check endpoint for the server."""
    global MODEL_CONFIG # Ensure we are referring to the global
    if MODEL_CONFIG: 
        response_data = {"status": "ok", "message": "Server is healthy, configuration loaded."}
        status_code = 200
    else:
        response_data = {"status": "error", "message": "Server is running, but configuration might have issues (e.g., no models loaded)."}
        status_code = 500
    
    resp = jsonify(response_data)
    resp.status_code = status_code
    return resp

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path: str | None = None): # path can be None
    global MODEL_CONFIG # Ensure we are referring to the global

    # Handle /v1/models endpoint directly here
    # Ensure path is not None before calling endswith
    if path and (path.endswith('/models') or path.endswith('/models/')):
        models_response = []
        if MODEL_CONFIG:
            for model_config_entry in MODEL_CONFIG:
                model_id = model_config_entry.get("model")
                provider = model_config_entry.get("provider", "unknown")
                if model_id:
                    models_response.append({
                        "id": model_id, "object": "model", "created": 1686935002, 
                        "owned_by": provider, "provider": provider,
                        "provider_model": model_config_entry.get("providerModel", "")
                    })
        
        resp_data = {"data": models_response, "object": "list"}
        return jsonify(resp_data)

    # --- Main Proxy Logic ---
    raw_data_bytes = request.get_data()
    
    try:
        decoded_data_str = raw_data_bytes.decode('utf-8') if raw_data_bytes else "{}"
        original_request_json = json.loads(decoded_data_str) if decoded_data_str else {}
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    if request.method != 'POST' or 'model' not in original_request_json:
        error_msg_detail = "POST request JSON body must include a 'model' field for this endpoint." \
                           if request.method == 'POST' else \
                           "This endpoint primarily proxies POST requests with a 'model' field."
        return jsonify({"error": error_msg_detail}), 400

    requested_model_id = original_request_json.get('model')
    print(f"Requested model ID: {requested_model_id}")

    api_config_result = get_target_api_config(requested_model_id, MODEL_CONFIG)

    if api_config_result.get("error"):
        error_msg = api_config_result["error"]
        print(f"Error getting API config: {error_msg}")
        return jsonify({"error": error_msg}), 400

    target_api_url = api_config_result["target_api_url"]
    target_api_key = api_config_result["target_api_key"]
    target_model_for_provider = api_config_result["target_model"]

    json_data_for_downstream_call = original_request_json.copy()
    if target_model_for_provider != requested_model_id:
        json_data_for_downstream_call['model'] = target_model_for_provider
    
    data_to_send_final_bytes = json.dumps(json_data_for_downstream_call).encode('utf-8') if json_data_for_downstream_call else raw_data_bytes
    is_stream = original_request_json.get('stream', False)

    if target_api_url == "anthropic_sdk":
        return handle_anthropic_sdk_request(
            json_data_for_sdk=json_data_for_downstream_call,
            target_model=target_model_for_provider,
            target_api_key=target_api_key,
            is_stream=is_stream,
            original_request_json_data=original_request_json
        )
    else:
        # Handle URL construction more intelligently
        # If the target_api_url already contains a path (like /private/chat/completions),
        # don't append the original request path
        if path and path.startswith('v1/'):
            # For requests to /v1/chat/completions, extract just the endpoint part
            endpoint_part = path[3:]  # Remove 'v1/' prefix
            downstream_url = f"{target_api_url.rstrip('/')}/{endpoint_part}"
        else:
            # For other paths, use them as-is
            downstream_path_segment = path.lstrip('/') if path else ''
            downstream_url = f"{target_api_url.rstrip('/')}/{downstream_path_segment}"
        print(f"Proxying REST request to: {downstream_url}")

        downstream_headers = {
            k: v for k, v in request.headers.items() 
            if k.lower() not in ['host', 'authorization', 'content-length', 'connection', 'user-agent']
        }
        downstream_headers['Host'] = target_api_url.split('//')[-1].split('/')[0]
        if target_api_key:
            downstream_headers['Authorization'] = f'Bearer {target_api_key}'

        return handle_rest_api_request(
            method=request.method,
            url=downstream_url,
            headers=downstream_headers,
            data_bytes=data_to_send_final_bytes, 
            is_stream=is_stream,
            original_request_json_data=original_request_json
        )

def run_server_main(): # Renamed to avoid conflict with any 'main' in cli.py if imported directly
    """Sets up and runs the Flask development server."""
    global MODEL_CONFIG
    try:
        config_data = load_config()
        MODEL_CONFIG = config_data.get('models', [])
        print(f"Loaded {len(MODEL_CONFIG)} models from config.")
        if not MODEL_CONFIG:
            print("Warning: No models found in configuration.")
    except Exception as e:
        print(f"Critical Error loading config: {e}. Server might not function correctly.")
        MODEL_CONFIG = []

    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Dolphin Logger server on port {port}...")
    print(f"Configuration loaded from: {get_config_path()}")
    print(f"Logs will be stored in: {get_logs_dir()}")   
    
    app.run(host='0.0.0.0', port=port, debug=False)
