"""
Claude Code provider implementation for Dolphin Logger.
"""

import json
import uuid
from datetime import datetime
from flask import Response, stream_with_context, jsonify
from typing import AsyncIterator, Dict, Any

from ..logging_utils import get_current_log_file, _should_log_request, log_lock


def handle_claude_code_request(
    json_data_for_sdk: dict,
    target_model: str,
    target_api_key: str | None,
    is_stream: bool,
    original_request_json_data: dict,
    claude_code_path: str | None = None,
    max_output_tokens: int | None = None
) -> Response:
    """
    Handles requests to Claude Code.
    
    Args:
        json_data_for_sdk: The request data formatted for Claude Code
        target_model: The target model name
        target_api_key: The API key (may not be used if using subscription)
        is_stream: Whether this is a streaming request
        original_request_json_data: The original client request for logging
        claude_code_path: Path to Claude Code executable
        max_output_tokens: Maximum output tokens for the model
        
    Returns:
        Flask Response object
    """
    print(f"DEBUG - Using Claude Code for request with model: {target_model}")

    try:
        # Import Claude Code utilities
        from path_to_claude_code.run import runClaudeCode
        from path_to_claude_code.message_filter import filterMessagesForClaudeCode
        
        # Extract messages and system prompt
        original_messages = json_data_for_sdk.get('messages', [])
        
        # Separate system messages
        system_messages = []
        non_system_messages = []
        
        for message in original_messages:
            if message.get('role') == 'system':
                system_messages.append(message.get('content', ''))
            else:
                non_system_messages.append(message)
        
        # Combine system messages
        system_prompt = '\n\n'.join(system_messages) if system_messages else ""
        
        # Also check for top-level system parameter
        if not system_prompt and json_data_for_sdk.get('system'):
            system_prompt = json_data_for_sdk.get('system')
        
        # Filter messages (removes unsupported content like images)
        filtered_messages = filterMessagesForClaudeCode(non_system_messages)
        
        # Determine if using Vertex
        use_vertex = json_data_for_sdk.get('use_vertex', False)
        
        # Get model ID
        model_id = target_model or "claude-sonnet-4-20250514"
        
        # Get max tokens
        max_tokens = max_output_tokens or json_data_for_sdk.get('max_tokens', 4096)
        
        # Start Claude Code process
        claude_process = runClaudeCode({
            'systemPrompt': system_prompt,
            'messages': filtered_messages,
            'path': claude_code_path,
            'modelId': model_id,
            'maxOutputTokens': max_tokens,
        })
        
        if is_stream:
            print(f"DEBUG - Creating streaming request to Claude Code: model={model_id}")
            
            def generate_claude_code_stream():
                response_content_parts = []
                log_entry = {'request': original_request_json_data, 'response': None}
                usage = {
                    'inputTokens': 0,
                    'outputTokens': 0,
                    'cacheReadTokens': 0,
                    'cacheWriteTokens': 0,
                    'totalCost': 0
                }
                is_paid_usage = True
                
                try:
                    for chunk in claude_process:
                        # Handle text chunks
                        if isinstance(chunk, str):
                            response_content_parts.append(chunk)
                            openai_compatible_chunk = {
                                "choices": [{
                                    "delta": {"content": chunk},
                                    "index": 0,
                                    "finish_reason": None
                                }],
                                "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                "model": model_id,
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                            continue
                        
                        # Handle system init
                        if chunk.get('type') == 'system' and chunk.get('subtype') == 'init':
                            is_paid_usage = chunk.get('apiKeySource') != 'none'
                            print(f"DEBUG - Claude Code initialized (paid: {is_paid_usage})")
                            continue
                        
                        # Handle assistant messages
                        if chunk.get('type') == 'assistant' and 'message' in chunk:
                            message = chunk['message']
                            
                            # Check for errors
                            if message.get('stop_reason') is not None:
                                content = message['content'][0] if message['content'] else None
                                if content and 'text' in content and content['text'].startswith('API Error'):
                                    error_text = content['text']
                                    print(f"ERROR - Claude Code API Error: {error_text}")
                                    raise Exception(error_text)
                            
                            # Process content
                            for content in message.get('content', []):
                                if content.get('type') == 'text':
                                    text = content.get('text', '')
                                    response_content_parts.append(text)
                                    openai_compatible_chunk = {
                                        "choices": [{
                                            "delta": {"content": text},
                                            "index": 0,
                                            "finish_reason": None
                                        }],
                                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                        "model": model_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp())
                                    }
                                    yield f"data: {json.dumps(openai_compatible_chunk)}\n\n".encode('utf-8')
                                elif content.get('type') == 'thinking':
                                    # Optionally handle thinking/reasoning blocks
                                    thinking = content.get('thinking', '')
                                    print(f"DEBUG - Reasoning: {thinking[:100]}...")
                            
                            # Update usage
                            usage['inputTokens'] += message.get('usage', {}).get('input_tokens', 0)
                            usage['outputTokens'] += message.get('usage', {}).get('output_tokens', 0)
                            usage['cacheReadTokens'] += message.get('usage', {}).get('cache_read_input_tokens', 0)
                            usage['cacheWriteTokens'] += message.get('usage', {}).get('cache_creation_input_tokens', 0)
                            continue
                        
                        # Handle result with cost
                        if chunk.get('type') == 'result' and 'result' in chunk:
                            usage['totalCost'] = chunk.get('total_cost_usd', 0) if is_paid_usage else 0
                            print(f"DEBUG - Usage: {usage}")
                    
                    # Send final chunk
                    final_chunk = {
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }],
                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "usage": {
                            "prompt_tokens": usage['inputTokens'],
                            "completion_tokens": usage['outputTokens'],
                            "total_tokens": usage['inputTokens'] + usage['outputTokens']
                        }
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"
                    
                finally:
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_entry['usage'] = usage
                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')
            
            resp = Response(stream_with_context(generate_claude_code_stream()), content_type='text/event-stream')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
        
        else:
            print(f"DEBUG - Creating non-streaming request to Claude Code: model={model_id}")
            
            # For non-streaming, collect all content
            response_content_parts = []
            usage = {
                'inputTokens': 0,
                'outputTokens': 0,
                'cacheReadTokens': 0,
                'cacheWriteTokens': 0
            }
            
            for chunk in claude_process:
                if isinstance(chunk, str):
                    response_content_parts.append(chunk)
                elif chunk.get('type') == 'assistant' and 'message' in chunk:
                    message = chunk['message']
                    for content in message.get('content', []):
                        if content.get('type') == 'text':
                            response_content_parts.append(content.get('text', ''))
                    
                    usage['inputTokens'] += message.get('usage', {}).get('input_tokens', 0)
                    usage['outputTokens'] += message.get('usage', {}).get('output_tokens', 0)
            
            response_content = "".join(response_content_parts)
            
            response_data = {
                "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": usage['inputTokens'],
                    "completion_tokens": usage['outputTokens'],
                    "total_tokens": usage['inputTokens'] + usage['outputTokens']
                }
            }
            
            print(f"DEBUG - Response from Claude Code received. Preview: {response_content[:100]}...")
            
            if _should_log_request(original_request_json_data):
                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps({
                            'request': original_request_json_data,
                            'response': response_content,
                            'usage': usage
                        }) + '\n')
            
            resp = jsonify(response_data)
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp
    
    except Exception as e:
        print(f"ERROR DETAILS (Claude Code): Type: {type(e).__name__}, Message: {str(e)}")
        error_body = {"error": {"message": str(e), "type": type(e).__name__}}
        resp = jsonify(error_body)
        resp.status_code = 500
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
