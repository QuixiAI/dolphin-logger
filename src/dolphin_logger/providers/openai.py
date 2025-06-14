"""
OpenAI provider implementation for Dolphin Logger.
Handles OpenAI and OpenAI-compatible APIs (including Ollama).
"""

import json
import requests
from flask import Response, stream_with_context, jsonify

from ..logging_utils import get_current_log_file, _should_log_request, log_lock


def handle_openai_request(
    method: str, 
    url: str, 
    headers: dict, 
    data_bytes: bytes, 
    is_stream: bool, 
    original_request_json_data: dict
) -> Response:
    """
    Handles requests to OpenAI-compatible (including Ollama) REST APIs.
    
    Args:
        method: HTTP method for the request
        url: Target API URL
        headers: Request headers
        data_bytes: Request body as bytes
        is_stream: Whether this is a streaming request
        original_request_json_data: The original client request for logging
        
    Returns:
        Flask Response object
    """
    print(f"DEBUG - Sending REST request: {method} {url}")

    try:
        api_response = requests.request(
            method=method, url=url, headers=headers, data=data_bytes,
            stream=is_stream, timeout=300
        )
        print(f"DEBUG - REST API Response status: {api_response.status_code}")
        
        api_response.raise_for_status()

        log_entry = {'request': original_request_json_data, 'response': None}

        if is_stream:
            def generate_rest_stream_response():
                response_content_parts = []
                try:
                    for line in api_response.iter_lines():
                        if line:
                            yield line + b'\n\n'
                            # For logging, accumulate content from delta if possible
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
                                        pass
                finally:
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')
            
            resp = Response(stream_with_context(generate_rest_stream_response()), 
                          content_type=api_response.headers.get('Content-Type', 'text/event-stream'))
            # Propagate relevant headers from upstream response
            for hdr_key in ['Cache-Control', 'Content-Type', 'Transfer-Encoding', 'Date', 'Server']:
                if hdr_key in api_response.headers: 
                    resp.headers[hdr_key] = api_response.headers[hdr_key]
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            complete_response_text = ''
            try:
                response_json = api_response.json()
                choices = response_json.get('choices', [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get('message', {})
                    if message and isinstance(message, dict): 
                        complete_response_text = message.get('content', '')
                if not complete_response_text and isinstance(response_json, dict):
                    complete_response_text = response_json
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
            resp = Response(api_response.content, 
                          content_type=api_response.headers.get('Content-Type', 'application/json'), 
                          status=api_response.status_code)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except requests.exceptions.RequestException as e:
        print(f"ERROR DETAILS (REST API - RequestException): Type: {type(e).__name__}, Message: {str(e)}, URL: {url}")
        error_content_type = 'application/json'
        error_status = 502  # Bad Gateway, typically for network issues with upstream
        error_body_msg = f"Error connecting to upstream API: {str(e)}"
        error_body = {"error": {"message": error_body_msg, "type": type(e).__name__}}

        if e.response is not None:
            print(f"  Upstream response status code: {e.response.status_code}")
            error_status = e.response.status_code
            error_content_type = e.response.headers.get('Content-Type', 'application/json')
            try: 
                error_body = e.response.json()
            except json.JSONDecodeError: 
                error_body = {"error": {"message": e.response.text, "type": "upstream_error"}}
        
        resp = jsonify(error_body)
        resp.status_code = error_status
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        print(f"UNEXPECTED REST API HANDLER ERROR: Type: {type(e).__name__}, Message: {str(e)}")
        resp = jsonify({"error": {"message": "An unexpected error occurred in the REST API handler.", "type": type(e).__name__}})
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
