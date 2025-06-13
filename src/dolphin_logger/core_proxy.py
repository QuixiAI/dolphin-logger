import json
import requests
import anthropic 
import uuid
from datetime import datetime
from flask import Response, stream_with_context, jsonify

from .logging_utils import get_current_log_file, _should_log_request, log_lock
# MODEL_CONFIG is typically loaded in server.py and passed to get_target_api_config.

def get_target_api_config(requested_model_id: str, model_config_list: list) -> dict:
    """
    Determines the target API URL, key, and model based on the requested model ID
    and the server's model configuration.

    Args:
        requested_model_id (str): The model ID from the client's request.
        model_config_list (list): The MODEL_CONFIG list (passed from server.py).

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
                    "target_api_key": "", 
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
                    # Ensure a specific error message for missing apiBase for OpenAI-like providers
                    return {"error": f"apiBase not configured for OpenAI-compatible model '{requested_model_id}'"}
                return {
                    "target_api_url": api_base,
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }

    if not model_config_list: # Check if the list itself is empty
        return {"error": f"No models configured. Cannot process request for model '{requested_model_id}'."}
    return {"error": f"Model '{requested_model_id}' not found in configured models."}


def handle_anthropic_sdk_request(
    json_data_for_sdk: dict, 
    target_model: str, 
    target_api_key: str | None, 
    is_stream: bool, 
    original_request_json_data: dict # This is the original client request for logging
) -> Response:
    """
    Handles requests to the Anthropic SDK.
    """
    print(f"DEBUG - Using Anthropic SDK for request with model: {target_model}")

    try:
        if not target_api_key: # Should be caught by config validation ideally, but good check.
            raise ValueError("Anthropic API key is missing in the configuration for the requested model.")
        
        # Ensure anthropic client is initialized correctly (as per its docs if specific settings needed)
        anthropic_client = anthropic.Anthropic(api_key=target_api_key)

        original_messages = json_data_for_sdk.get('messages', [])
        max_tokens = json_data_for_sdk.get('max_tokens', 4096) # Default from Anthropic
        
        # Extract system messages from the messages array and separate them
        system_messages = []
        non_system_messages = []
        
        for message in original_messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
            else:
                non_system_messages.append(message)
        
        # Combine system messages into a single system prompt
        system_prompt = '\n\n'.join(system_messages) if system_messages else None
        
        # Also check for top-level system parameter (in case it's already properly formatted)
        if not system_prompt and json_data_for_sdk.get('system'):
            system_prompt = json_data_for_sdk.get('system')

        anthropic_args = {
            "model": target_model, # This is the providerModel
            "messages": non_system_messages,  # Only non-system messages
            "max_tokens": max_tokens,
            "stream": is_stream,
        }
        if system_prompt: # Only add system to args if it's present
            anthropic_args["system"] = system_prompt

        if is_stream:
            print(f"DEBUG - Creating streaming request to Anthropic API: model={target_model}")
            sdk_stream = anthropic_client.messages.create(**anthropic_args)

            def generate_anthropic_stream_response():
                response_content_parts = []
                # Use original_request_json_data for logging the request part
                log_entry = {'request': original_request_json_data, 'response': None}
                try:
                    for chunk in sdk_stream:
                        if chunk.type == "content_block_delta":
                            delta_content = chunk.delta.text
                            response_content_parts.append(delta_content)
                            openai_compatible_chunk = {
                                "choices": [{"delta": {"content": delta_content}, "index": 0, "finish_reason": None}],
                                "id": f"chatcmpl-anthropic-{uuid.uuid4()}", "model": target_model, 
                                "object": "chat.completion.chunk", "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                        elif chunk.type == "message_stop":
                            finish_reason = chunk.message.stop_reason if hasattr(chunk, 'message') and hasattr(chunk.message, 'stop_reason') else "stop"
                            final_chunk = {
                                 "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
                                "id": f"chatcmpl-anthropic-{uuid.uuid4()}", "model": target_model,
                                "object": "chat.completion.chunk", "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                finally: # Ensure logging happens even if stream breaks or client disconnects
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')

            resp = Response(stream_with_context(generate_anthropic_stream_response()), content_type='text/event-stream')
            resp.headers['Access-Control-Allow-Origin'] = '*' # CORS for streaming response
            return resp
        else: # Non-streaming Anthropic request
            print(f"DEBUG - Creating non-streaming request to Anthropic API: model={target_model}")
            response_obj = anthropic_client.messages.create(**anthropic_args)
            
            response_content = response_obj.content[0].text if response_obj.content and len(response_obj.content) > 0 and hasattr(response_obj.content[0], 'text') else ""
            
            response_data_converted = {
                "id": f"chatcmpl-anthropic-{response_obj.id}", "object": "chat.completion",
                "created": int(datetime.now().timestamp()), # Consider more accurate timestamp if available
                "model": response_obj.model, # Use model from Anthropic's response
                "choices": [{"index": 0, "message": {"role": response_obj.role, "content": response_content}, "finish_reason": response_obj.stop_reason }],
                "usage": {"prompt_tokens": response_obj.usage.input_tokens, "completion_tokens": response_obj.usage.output_tokens, "total_tokens": response_obj.usage.input_tokens + response_obj.usage.output_tokens}
            }
            print(f"DEBUG - Response from Anthropic API received. Preview: {response_content[:100]}...")
            if _should_log_request(original_request_json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps({'request': original_request_json_data, 'response': response_content}) + '\n')
            
            resp = jsonify(response_data_converted)
            resp.headers['Access-Control-Allow-Origin'] = '*' # CORS for non-streaming response
            return resp

    except anthropic.APIError as e: # Specific error handling for Anthropic
        # Extract error details safely - BadRequestError and other APIError subclasses may not have 'type' attribute
        error_type = getattr(e, 'type', type(e).__name__)
        error_message = getattr(e, 'message', str(e))
        status_code = getattr(e, 'status_code', 500)
        
        print(f"ERROR DETAILS (Anthropic SDK - APIError): Status: {status_code}, Type: {error_type}, Message: {error_message}")
        
        error_body = {"error": {"message": error_message, "type": error_type, "code": status_code}}
        resp = jsonify(error_body)
        resp.status_code = status_code
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e: # Catch-all for other errors (e.g., ValueError, network issues not caught by APIError)
        print(f"ERROR DETAILS (Anthropic SDK - General): Type: {type(e).__name__}, Message: {str(e)}")
        if original_request_json_data: # Log original request for context on error
            print(f"  Full request data (original for logging): {json.dumps(original_request_json_data)}")
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}
        resp = jsonify(error_body)
        resp.status_code = 500 # Internal Server Error for unexpected issues
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp


def handle_rest_api_request(
    method: str, 
    url: str, 
    headers: dict, 
    data_bytes: bytes, # This is the already encoded byte string for the request body
    is_stream: bool, 
    original_request_json_data: dict # This is the original client request (parsed JSON) for logging
) -> Response:
    """
    Handles requests to OpenAI-compatible (including Ollama) REST APIs.
    """
    print(f"DEBUG - Sending REST request: {method} {url}")
    # Avoid logging full headers/data here in production due to sensitivity. Previews are safer.
    # print(f"DEBUG - Headers: {headers}")
    # if data_bytes: print(f"DEBUG - Request data preview: {data_bytes[:200]}...")

    try:
        api_response = requests.request(
            method=method, url=url, headers=headers, data=data_bytes,
            stream=is_stream, timeout=300 # Standard timeout
        )
        print(f"DEBUG - REST API Response status: {api_response.status_code}")
        # if not is_stream: print(f"DEBUG - REST API Response preview: {api_response.text[:200]}...")
        
        api_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        log_entry = {'request': original_request_json_data, 'response': None}

        if is_stream:
            def generate_rest_stream_response():
                response_content_parts = []
                try:
                    for line in api_response.iter_lines(): # iter_lines decodes by default (utf-8)
                        if line:
                            yield line + b'\n\n' # Pass through to client, maintaining SSE format
                            # For logging, accumulate content from delta if possible
                            if line.startswith(b'data: '):
                                line_data_str = line.decode('utf-8', errors='replace')[6:] # Get content part
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
                                        # print(f"Warning: Error processing stream line for REST logging: {line_data_str}")
                                        pass # Don't let logging error break the stream to client
                finally: # Ensure logging happens
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')
            
            resp = Response(stream_with_context(generate_rest_stream_response()), content_type=api_response.headers.get('Content-Type', 'text/event-stream'))
            # Propagate relevant headers from upstream response
            for hdr_key in ['Cache-Control', 'Content-Type', 'Transfer-Encoding', 'Date', 'Server']:
                if hdr_key in api_response.headers: resp.headers[hdr_key] = api_response.headers[hdr_key]
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else: # Non-streaming REST response
            complete_response_text = ''
            try:
                response_json = api_response.json() # Parse the JSON from the target API response
                choices = response_json.get('choices', [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get('message', {})
                    if message and isinstance(message, dict): complete_response_text = message.get('content', '')
                if not complete_response_text and isinstance(response_json, dict): # If content extraction failed, log whole JSON
                    complete_response_text = response_json # Log the full JSON object
                log_entry['response'] = complete_response_text
            except json.JSONDecodeError:
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
        print(f"ERROR DETAILS (REST API - RequestException): Type: {type(e).__name__}, Message: {str(e)}, URL: {url}")
        error_content_type = 'application/json'
        error_status = 502 # Bad Gateway, typically for network issues with upstream
        error_body_msg = f"Error connecting to upstream API: {str(e)}"
        error_body = {"error": {"message": error_body_msg, "type": type(e).__name__}}

        if e.response is not None: # If the exception is from an HTTP error status (4xx, 5xx from upstream)
            print(f"  Upstream response status code: {e.response.status_code}")
            error_status = e.response.status_code # Use upstream's status code
            error_content_type = e.response.headers.get('Content-Type', 'application/json')
            try: error_body = e.response.json() # Try to parse upstream error
            except json.JSONDecodeError: error_body = {"error": {"message": e.response.text, "type": "upstream_error"}}
        
        resp = jsonify(error_body)
        resp.status_code = error_status
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e: # Catch-all for any other unexpected errors
        print(f"UNEXPECTED REST API HANDLER ERROR: Type: {type(e).__name__}, Message: {str(e)}")
        # import traceback; traceback.print_exc() # Good for debugging
        resp = jsonify({"error": {"message": "An unexpected error occurred in the REST API handler.", "type": type(e).__name__}})
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
