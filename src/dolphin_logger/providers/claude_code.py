"""
Claude Code provider implementation for Dolphin Logger.

This provider uses Claude Code CLI to leverage Claude Max subscriptions
while still capturing detailed API interaction logs.
"""

import json
import uuid
import subprocess
import os
from datetime import datetime
from flask import Response, stream_with_context, jsonify
from typing import Iterator, Dict, Any

from ..logging_utils import get_current_log_file, _should_log_request, log_lock


def _convert_to_claude_messages(openai_messages: list) -> tuple[str, list]:
    """
    Convert OpenAI format messages to Claude format.

    Returns:
        Tuple of (system_prompt, messages_list)
    """
    system_parts = []
    messages = []

    for msg in openai_messages:
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'system':
            system_parts.append(content)
        elif role in ['user', 'assistant']:
            # Convert content to Claude format
            if isinstance(content, str):
                messages.append({
                    'role': role,
                    'content': content
                })
            elif isinstance(content, list):
                # Handle multi-part content (text + images)
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    # Note: Images are not supported by Claude Code CLI

                if text_parts:
                    messages.append({
                        'role': role,
                        'content': '\n'.join(text_parts)
                    })

    system_prompt = '\n\n'.join(system_parts) if system_parts else ''
    return system_prompt, messages


def _invoke_claude_code(
    messages: list,
    system_prompt: str = '',
    claude_code_path: str | None = None,
    claude_code_oauth_token: str | None = None,
    max_tokens: int | None = None
) -> Iterator[Dict[str, Any]]:
    """
    Invoke Claude Code CLI and stream the response as structured chunks.

    Note: Claude Code handles authentication and model selection internally,
    but outputs detailed API interaction data as JSON chunks.

    Yields structured dictionaries containing:
    - Text chunks: {"type": "text", "text": "..."}
    - Usage chunks: {"type": "usage", "usage": {...}}
    - System chunks: {"type": "system", "data": {...}}
    - Cost chunks: {"type": "cost", "cost_usd": 0.123}
    - etc.
    """
    # Find Claude Code executable
    if claude_code_path and os.path.exists(claude_code_path):
        cmd = [claude_code_path]
    else:
        # Try to find in PATH
        cmd = ['claude']

    # Build the command arguments
    cmd.extend(['chat'])

    # Use print mode for non-interactive output
    cmd.append('--print')

    # Use JSON streaming output format (requires --verbose)
    cmd.extend(['--output-format', 'stream-json'])
    cmd.append('--verbose')

    # Only add max-tokens if explicitly specified
    if max_tokens:
        cmd.extend(['--max-tokens', str(max_tokens)])

    # Add system prompt if provided
    if system_prompt:
        cmd.extend(['--system-prompt', system_prompt])

    # Prepare input - convert to Claude Code's expected format
    # Build the user message from the messages
    user_parts = []

    for msg in messages:
        if msg['role'] == 'user':
            user_parts.append(msg['content'])
        elif msg['role'] == 'assistant':
            # Include assistant messages in context
            user_parts.append(f"[Assistant previously said: {msg['content']}]")

    user_message = '\n\n'.join(user_parts)

    try:

        # Set up environment with OAuth token if provided
        env = os.environ.copy()
        if claude_code_oauth_token:
            env['CLAUDE_CODE_OAUTH_TOKEN'] = claude_code_oauth_token

        # Run Claude Code (uses its own cached authentication)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )

        # Send the user message to stdin
        if process.stdin:
            process.stdin.write(user_message)
            process.stdin.close()

        # Stream output line by line, parsing JSON chunks
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON chunk from Claude Code
                    chunk = json.loads(line)
                    yield chunk
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text output
                    print(f"DEBUG - Non-JSON output from Claude Code: {line}")
                    yield {"type": "text", "text": line}

        # Wait for process to complete
        return_code = process.wait()

        # Check for errors
        if return_code != 0:
            error_output = process.stderr.read() if process.stderr else ""
            print(f"ERROR - Claude Code failed with code {return_code}: {error_output}")
            raise Exception(f"Claude Code error (code {return_code}): {error_output}")

    except FileNotFoundError:
        raise Exception(
            "Claude Code executable not found. Please install Claude Code or "
            "specify the correct path in 'claudeCodePath' config."
        )
    except Exception as e:
        print(f"ERROR - Failed to invoke Claude Code: {e}")
        raise


def handle_claude_code_request(
    json_data_for_sdk: dict,
    target_model: str,
    target_api_key: str | None,
    is_stream: bool,
    original_request_json_data: dict,
    claude_code_path: str | None = None,
    claude_code_oauth_token: str | None = None,
    max_output_tokens: int | None = None
) -> Response:
    """
    Handles requests to Claude Code.

    Args:
        json_data_for_sdk: The request data formatted for Claude Code
        target_model: Not used (Claude Code handles model selection internally)
        target_api_key: Not used (Claude Code handles auth internally)
        is_stream: Whether this is a streaming request
        original_request_json_data: The original client request for logging
        claude_code_path: Path to Claude Code executable
        max_output_tokens: Optional maximum output tokens override

    Returns:
        Flask Response object
    """
    print(f"DEBUG - Using Claude Code for request")

    try:
        # Extract messages
        original_messages = json_data_for_sdk.get('messages', [])

        # Also check for top-level system parameter
        top_level_system = json_data_for_sdk.get('system', '')

        # Convert messages
        system_prompt, messages = _convert_to_claude_messages(original_messages)

        # Prefer top-level system if provided
        if top_level_system:
            system_prompt = top_level_system

        # Get max tokens if specified
        max_tokens = max_output_tokens or json_data_for_sdk.get('max_tokens')

        if is_stream:
            print(f"DEBUG - Creating streaming request to Claude Code")

            def generate_claude_code_stream():
                response_content_parts = []
                usage_data = {
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'cache_creation_input_tokens': 0
                }
                cost_usd = 0
                is_paid_usage = True

                log_entry = {
                    'request': original_request_json_data,
                    'response': None,
                    'usage': None,
                    'cost_usd': None,
                    'api_key_source': None
                }

                try:
                    for chunk in _invoke_claude_code(
                        messages=messages,
                        system_prompt=system_prompt,
                        claude_code_path=claude_code_path,
                        claude_code_oauth_token=claude_code_oauth_token,
                        max_tokens=max_tokens
                    ):
                        # Handle different chunk types from Claude Code
                        chunk_type = chunk.get('type')

                        if chunk_type == 'system' and chunk.get('subtype') == 'init':
                            # System initialization - check if using subscription or paid API
                            api_key_source = chunk.get('apiKeySource', 'unknown')
                            is_paid_usage = api_key_source != 'none'
                            log_entry['api_key_source'] = api_key_source
                            print(f"DEBUG - Claude Code initialized (API key source: {api_key_source})")
                            continue

                        elif chunk_type == 'assistant' and 'message' in chunk:
                            # Assistant message with content and usage
                            message = chunk['message']

                            # Check for API errors
                            if message.get('stop_reason') is not None:
                                content = message['content'][0] if message.get('content') else None
                                if content and 'text' in content and content['text'].startswith('API Error'):
                                    error_text = content['text']
                                    print(f"ERROR - Claude Code API Error: {error_text}")
                                    raise Exception(error_text)

                            # Process content blocks
                            for content in message.get('content', []):
                                if content.get('type') == 'text':
                                    text = content.get('text', '')
                                    response_content_parts.append(text)

                                    # Send text as streaming chunk
                                    openai_chunk = {
                                        "choices": [{
                                            "delta": {"content": text},
                                            "index": 0,
                                            "finish_reason": None
                                        }],
                                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                        "model": "claude-code",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp())
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                                elif content.get('type') == 'tool_use':
                                    # Format tool use as readable text
                                    tool_name = content.get('name', 'unknown')
                                    tool_input = content.get('input', {})
                                    tool_text = f"\n[Tool: {tool_name}]\n{json.dumps(tool_input, indent=2)}\n"
                                    response_content_parts.append(tool_text)

                                    # Send tool use as text chunk
                                    openai_chunk = {
                                        "choices": [{
                                            "delta": {"content": tool_text},
                                            "index": 0,
                                            "finish_reason": None
                                        }],
                                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                        "model": "claude-code",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp())
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                                elif content.get('type') == 'tool_result':
                                    # Format tool result as readable text
                                    tool_result = content.get('content', '')
                                    if isinstance(tool_result, list):
                                        tool_result = '\n'.join([str(r) for r in tool_result])
                                    result_text = f"\n[Tool Result]\n{tool_result}\n"
                                    response_content_parts.append(result_text)

                                    # Send tool result as text chunk
                                    openai_chunk = {
                                        "choices": [{
                                            "delta": {"content": result_text},
                                            "index": 0,
                                            "finish_reason": None
                                        }],
                                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                        "model": "claude-code",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp())
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                                elif content.get('type') == 'thinking':
                                    # Include thinking in response as well
                                    thinking = content.get('thinking', '')
                                    thinking_text = f"\n[Thinking]\n{thinking}\n"
                                    response_content_parts.append(thinking_text)

                                    # Send thinking as text chunk
                                    openai_chunk = {
                                        "choices": [{
                                            "delta": {"content": thinking_text},
                                            "index": 0,
                                            "finish_reason": None
                                        }],
                                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                        "model": "claude-code",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp())
                                    }
                                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')
                                    print(f"DEBUG - Thinking block: {thinking[:100]}...")

                            # Accumulate usage
                            msg_usage = message.get('usage', {})
                            usage_data['input_tokens'] += msg_usage.get('input_tokens', 0)
                            usage_data['output_tokens'] += msg_usage.get('output_tokens', 0)
                            usage_data['cache_read_input_tokens'] += msg_usage.get('cache_read_input_tokens', 0)
                            usage_data['cache_creation_input_tokens'] += msg_usage.get('cache_creation_input_tokens', 0)

                        elif chunk_type == 'result':
                            # Final result with cost
                            # Check for errors
                            if chunk.get('is_error'):
                                error_msg = chunk.get('result', 'Unknown error')
                                print(f"ERROR - Claude Code returned error: {error_msg}")
                                raise Exception(f"Claude Code error: {error_msg}")

                            cost_usd = chunk.get('total_cost_usd', 0) if is_paid_usage else 0
                            log_entry['cost_usd'] = cost_usd
                            print(f"DEBUG - Total cost: ${cost_usd:.4f}")

                        elif chunk_type == 'text':
                            # Plain text chunk
                            text = chunk.get('text', '')
                            response_content_parts.append(text)

                            openai_chunk = {
                                "choices": [{
                                    "delta": {"content": text},
                                    "index": 0,
                                    "finish_reason": None
                                }],
                                "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                                "model": "claude-code",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp())
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                    # Send final chunk with usage
                    final_chunk = {
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }],
                        "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                        "model": "claude-code",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "usage": {
                            "prompt_tokens": usage_data['input_tokens'],
                            "completion_tokens": usage_data['output_tokens'],
                            "total_tokens": usage_data['input_tokens'] + usage_data['output_tokens'],
                            "cache_read_tokens": usage_data['cache_read_input_tokens'],
                            "cache_creation_tokens": usage_data['cache_creation_input_tokens']
                        }
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8')
                    yield b"data: [DONE]\n\n"

                finally:
                    # Log the complete interaction
                    if _should_log_request(original_request_json_data):
                        log_entry['response'] = "".join(response_content_parts)
                        log_entry['usage'] = usage_data

                        log_file_path = get_current_log_file()
                        with log_lock:
                            with open(log_file_path, 'a') as log_file:
                                log_file.write(json.dumps(log_entry) + '\n')

                        print(f"DEBUG - Logged interaction: {len(response_content_parts)} parts, "
                              f"{usage_data['input_tokens']} input tokens, "
                              f"{usage_data['output_tokens']} output tokens, "
                              f"${cost_usd:.4f} cost")

            resp = Response(stream_with_context(generate_claude_code_stream()), content_type='text/event-stream')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp

        else:
            print(f"DEBUG - Creating non-streaming request to Claude Code")

            # For non-streaming, collect all content
            response_content_parts = []
            usage_data = {
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_read_input_tokens': 0,
                'cache_creation_input_tokens': 0
            }
            cost_usd = 0
            is_paid_usage = True
            api_key_source = None

            for chunk in _invoke_claude_code(
                messages=messages,
                system_prompt=system_prompt,
                claude_code_path=claude_code_path,
                claude_code_oauth_token=claude_code_oauth_token,
                max_tokens=max_tokens
            ):
                chunk_type = chunk.get('type')

                if chunk_type == 'system' and chunk.get('subtype') == 'init':
                    api_key_source = chunk.get('apiKeySource', 'unknown')
                    is_paid_usage = api_key_source != 'none'

                elif chunk_type == 'assistant' and 'message' in chunk:
                    message = chunk['message']

                    for content in message.get('content', []):
                        if content.get('type') == 'text':
                            response_content_parts.append(content.get('text', ''))
                        elif content.get('type') == 'tool_use':
                            # Format tool use as readable text
                            tool_name = content.get('name', 'unknown')
                            tool_input = content.get('input', {})
                            tool_text = f"\n[Tool: {tool_name}]\n{json.dumps(tool_input, indent=2)}\n"
                            response_content_parts.append(tool_text)
                        elif content.get('type') == 'tool_result':
                            # Format tool result as readable text
                            tool_result = content.get('content', '')
                            if isinstance(tool_result, list):
                                tool_result = '\n'.join([str(r) for r in tool_result])
                            result_text = f"\n[Tool Result]\n{tool_result}\n"
                            response_content_parts.append(result_text)
                        elif content.get('type') == 'thinking':
                            # Include thinking in response
                            thinking = content.get('thinking', '')
                            thinking_text = f"\n[Thinking]\n{thinking}\n"
                            response_content_parts.append(thinking_text)

                    msg_usage = message.get('usage', {})
                    usage_data['input_tokens'] += msg_usage.get('input_tokens', 0)
                    usage_data['output_tokens'] += msg_usage.get('output_tokens', 0)
                    usage_data['cache_read_input_tokens'] += msg_usage.get('cache_read_input_tokens', 0)
                    usage_data['cache_creation_input_tokens'] += msg_usage.get('cache_creation_input_tokens', 0)

                elif chunk_type == 'result':
                    # Check for errors
                    if chunk.get('is_error'):
                        error_msg = chunk.get('result', 'Unknown error')
                        raise Exception(f"Claude Code error: {error_msg}")

                    cost_usd = chunk.get('total_cost_usd', 0) if is_paid_usage else 0

                elif chunk_type == 'text':
                    response_content_parts.append(chunk.get('text', ''))

            response_content = "".join(response_content_parts)

            response_data = {
                "id": f"chatcmpl-claudecode-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": "claude-code",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": usage_data['input_tokens'],
                    "completion_tokens": usage_data['output_tokens'],
                    "total_tokens": usage_data['input_tokens'] + usage_data['output_tokens'],
                    "cache_read_tokens": usage_data['cache_read_input_tokens'],
                    "cache_creation_tokens": usage_data['cache_creation_input_tokens']
                }
            }

            print(f"DEBUG - Response from Claude Code: {len(response_content)} chars, "
                  f"{usage_data['input_tokens']} input tokens, "
                  f"{usage_data['output_tokens']} output tokens, "
                  f"${cost_usd:.4f} cost")

            # Log the interaction
            if _should_log_request(original_request_json_data):
                log_entry = {
                    'request': original_request_json_data,
                    'response': response_content,
                    'usage': usage_data,
                    'cost_usd': cost_usd,
                    'api_key_source': api_key_source
                }

                log_file_path = get_current_log_file()
                with log_lock:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(json.dumps(log_entry) + '\n')

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
