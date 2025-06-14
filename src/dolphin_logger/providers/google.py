"""
Google provider implementation for Dolphin Logger.
Handles Google AI APIs (Gemini, etc.) using the Google GenAI SDK.
"""

import json
import uuid
from datetime import datetime
from flask import Response, stream_with_context, jsonify

from ..logging_utils import get_current_log_file, _should_log_request, log_lock


def handle_google_request(
    json_data_for_sdk: dict, 
    target_model: str, 
    target_api_key: str | None, 
    is_stream: bool, 
    original_request_json_data: dict
) -> Response:
    """
    Handles requests to Google AI APIs using the Google GenAI SDK.
    
    Args:
        json_data_for_sdk: The request data formatted for the Google GenAI SDK
        target_model: The target model name
        target_api_key: The API key for Google GenAI
        is_stream: Whether this is a streaming request
        original_request_json_data: The original client request for logging
        
    Returns:
        Flask Response object
    """
    print(f"DEBUG - Using Google GenAI SDK for request with model: {target_model}")

    try:
        # Import Google GenAI SDK
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            error_msg = "Google GenAI SDK not installed. Please install with: pip install google-genai"
            print(f"ERROR: {error_msg}")
            resp = jsonify({"error": {"message": error_msg, "type": "import_error"}})
            resp.status_code = 500
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        if not target_api_key:
            raise ValueError("Google API key is missing in the configuration for the requested model.")
        
        # Initialize Google GenAI client
        client = genai.Client(api_key=target_api_key)

        # Convert OpenAI-format messages to Google GenAI format
        original_messages = json_data_for_sdk.get('messages', [])
        contents = []
        
        for message in original_messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Map OpenAI roles to Google GenAI roles
            if role == 'system':
                # Google GenAI doesn't have system role, so we'll prepend it to the first user message
                # or create a user message if none exists
                if contents and contents[-1].role == 'user':
                    # Prepend to last user message
                    existing_text = contents[-1].parts[0].text if contents[-1].parts else ""
                    new_text = f"{content}\n\n{existing_text}"
                    contents[-1] = types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=new_text)]
                    )
                else:
                    # Create new user message with system content
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=content)]
                    ))
            elif role == 'assistant':
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                ))
            else:  # user or other roles
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                ))

        # Configure safety settings to be permissive (similar to the example)
        generate_content_config = types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
            response_mime_type="text/plain",
        )

        if is_stream:
            print(f"DEBUG - Creating streaming request to Google GenAI API: model={target_model}")

            def generate_google_stream_response():
                response_content_parts = []
                log_entry = {'request': original_request_json_data, 'response': None}
                try:
                    for chunk in client.models.generate_content_stream(
                        model=target_model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if hasattr(chunk, 'text') and chunk.text:
                            delta_content = chunk.text
                            response_content_parts.append(delta_content)
                            
                            # Convert to OpenAI-compatible streaming format
                            openai_compatible_chunk = {
                                "choices": [{"delta": {"content": delta_content}, "index": 0, "finish_reason": None}],
                                "id": f"chatcmpl-google-{uuid.uuid4()}", 
                                "model": target_model, 
                                "object": "chat.completion.chunk", 
                                "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                    
                    # Send final chunk
                    final_chunk = {
                        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                        "id": f"chatcmpl-google-{uuid.uuid4()}", 
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

            resp = Response(stream_with_context(generate_google_stream_response()), content_type='text/event-stream')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        else:
            print(f"DEBUG - Creating non-streaming request to Google GenAI API: model={target_model}")
            
            # For non-streaming, we need to collect all the content
            response_parts = []
            for chunk in client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=generate_content_config,
            ):
                if hasattr(chunk, 'text') and chunk.text:
                    response_parts.append(chunk.text)
            
            response_content = "".join(response_parts)
            
            # Convert to OpenAI-compatible format
            response_data_converted = {
                "id": f"chatcmpl-google-{uuid.uuid4()}", 
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": target_model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Google GenAI doesn't provide token counts in the same way
            }
            
            print(f"DEBUG - Response from Google GenAI API received. Preview: {response_content[:100]}...")
            
            if _should_log_request(original_request_json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps({'request': original_request_json_data, 'response': response_content}) + '\n')
            
            resp = jsonify(response_data_converted)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

    except Exception as e:
        print(f"ERROR DETAILS (Google GenAI SDK): Type: {type(e).__name__}, Message: {str(e)}")
        if original_request_json_data:
            print(f"  Full request data (original for logging): {json.dumps(original_request_json_data)}")
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}
        resp = jsonify(error_body)
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
