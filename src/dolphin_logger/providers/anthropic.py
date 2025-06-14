"""
Anthropic provider implementation for Dolphin Logger.
"""

import json
import anthropic
import uuid
from datetime import datetime
from flask import Response, stream_with_context, jsonify

from ..logging_utils import get_current_log_file, _should_log_request, log_lock


def handle_anthropic_request(
    json_data_for_sdk: dict, 
    target_model: str, 
    target_api_key: str | None, 
    is_stream: bool, 
    original_request_json_data: dict
) -> Response:
    """
    Handles requests to the Anthropic SDK.
    
    Args:
        json_data_for_sdk: The request data formatted for the Anthropic SDK
        target_model: The target model name
        target_api_key: The API key for Anthropic
        is_stream: Whether this is a streaming request
        original_request_json_data: The original client request for logging
        
    Returns:
        Flask Response object
    """
    print(f"DEBUG - Using Anthropic SDK for request with model: {target_model}")

    try:
        if not target_api_key:
            raise ValueError("Anthropic API key is missing in the configuration for the requested model.")
        
        # Initialize Anthropic client
        anthropic_client = anthropic.Anthropic(api_key=target_api_key)

        original_messages = json_data_for_sdk.get('messages', [])
        max_tokens = json_data_for_sdk.get('max_tokens', 4096)
        
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
        
        # Also check for top-level system parameter
        if not system_prompt and json_data_for_sdk.get('system'):
            system_prompt = json_data_for_sdk.get('system')

        anthropic_args = {
            "model": target_model,
            "messages": non_system_messages,
            "max_tokens": max_tokens,
            "stream": is_stream,
        }
        if system_prompt:
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
        else:
            print(f"DEBUG - Creating non-streaming request to Anthropic API: model={target_model}")
            response_obj = anthropic_client.messages.create(**anthropic_args)
            
            response_content = response_obj.content[0].text if response_obj.content and len(response_obj.content) > 0 and hasattr(response_obj.content[0], 'text') else ""
            
            response_data_converted = {
                "id": f"chatcmpl-anthropic-{response_obj.id}", 
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": response_obj.model,
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
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except anthropic.APIError as e:
        error_type = getattr(e, 'type', type(e).__name__)
        error_message = getattr(e, 'message', str(e))
        status_code = getattr(e, 'status_code', 500)
        
        print(f"ERROR DETAILS (Anthropic SDK - APIError): Status: {status_code}, Type: {error_type}, Message: {error_message}")
        
        error_body = {"error": {"message": error_message, "type": error_type, "code": status_code}}
        resp = jsonify(error_body)
        resp.status_code = status_code
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        print(f"ERROR DETAILS (Anthropic SDK - General): Type: {type(e).__name__}, Message: {str(e)}")
        if original_request_json_data:
            print(f"  Full request data (original for logging): {json.dumps(original_request_json_data)}")
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}
        resp = jsonify(error_body)
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
