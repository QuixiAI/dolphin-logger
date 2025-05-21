import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
from pathlib import Path
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import functions to be tested from the new core_proxy module
from dolphin_logger.core_proxy import get_target_api_config, handle_anthropic_sdk_request, handle_rest_api_request
# Import other necessary components that might be used by these functions or their mocks
from dolphin_logger.logging_utils import log_lock # log_lock is used in handlers
from flask import Flask # For creating response objects in mocks, or if handlers return Flask responses

# Common test data (can be refactored if needed)
MOCK_ANTHROPIC_KEY_CORE = "fake_anthropic_key_core_test"
MOCK_OPENAI_KEY_CORE = "fake_openai_key_core_test"
MOCK_ENV_KEY_NAME_CORE = "MY_CORE_TEST_KEY"
MOCK_ENV_KEY_VALUE_CORE = "resolved_core_env_key"

@pytest.fixture
def mock_model_config_for_core():
    """Model configuration fixture for core_proxy tests."""
    return [
        {
            "provider": "openai", "providerModel": "gpt-4-core", "model": "openai-gpt4-core",
            "apiBase": "https://api.openai.com/v1", "apiKey": MOCK_OPENAI_KEY_CORE
        },
        {
            "provider": "anthropic", "providerModel": "claude-3-opus-core", "model": "anthropic-claude3-core",
            "apiKey": MOCK_ANTHROPIC_KEY_CORE
        },
        {
            "provider": "ollama", "providerModel": "llama3-core", "model": "ollama-llama3-core",
            "apiBase": "http://localhost:11634/v1" # Yet another port
        },
        {
            "provider": "openai", "providerModel": "gpt-3.5-turbo-core-env", "model": "openai-gpt3.5-core-env",
            "apiBase": "https://api.openai.com/v1", "apiKey": f"ENV:{MOCK_ENV_KEY_NAME_CORE}"
        }
    ]

@pytest.fixture
def mock_model_config_empty_for_core():
    return []

# --- Tests for get_target_api_config (formerly _get_target_api_config) ---
@pytest.mark.parametrize("requested_model_id, expected_provider, expected_api_url_contains, expected_key_or_resolved, expected_target_model_name", [
    ("openai-gpt4-core", "openai", "api.openai.com/v1", MOCK_OPENAI_KEY_CORE, "gpt-4-core"),
    ("anthropic-claude3-core", "anthropic", "anthropic_sdk", MOCK_ANTHROPIC_KEY_CORE, "claude-3-opus-core"),
    ("ollama-llama3-core", "ollama", "localhost:11634/v1", "", "llama3-core"),
    ("openai-gpt3.5-core-env", "openai", "api.openai.com/v1", MOCK_ENV_KEY_VALUE_CORE, "gpt-3.5-turbo-core-env"),
])
def test_get_target_api_config_found_core(
    mock_model_config_for_core, requested_model_id, expected_provider, 
    expected_api_url_contains, expected_key_or_resolved, expected_target_model_name, mocker
):
    # Simulate ENV var resolution if testing that case
    # Note: In the actual app, load_config in config.py does this resolution.
    # get_target_api_config receives the already-resolved config.
    # So, we need to ensure the mock_model_config_for_core reflects this for the ENV var case.
    active_model_config = mock_model_config_for_core
    if requested_model_id == "openai-gpt3.5-core-env":
        # Mock os.environ for this specific test, as if load_config was called
        mocker.patch.dict(os.environ, {MOCK_ENV_KEY_NAME_CORE: MOCK_ENV_KEY_VALUE_CORE})
        # Manually "resolve" it in our test fixture for this test run
        # This simulates what load_config would do before passing MODEL_CONFIG
        for m_cfg in active_model_config:
            if m_cfg["model"] == "openai-gpt3.5-core-env":
                m_cfg["apiKey"] = MOCK_ENV_KEY_VALUE_CORE 
    
    # Pass the (potentially modified for ENV var) model config list
    config_result = get_target_api_config(requested_model_id, active_model_config)
    
    assert config_result["error"] is None
    assert config_result["provider"] == expected_provider
    assert expected_api_url_contains in config_result["target_api_url"]
    assert config_result["target_api_key"] == expected_key_or_resolved
    assert config_result["target_model"] == expected_target_model_name

def test_get_target_api_config_not_found_core(mock_model_config_for_core):
    config_result = get_target_api_config("non-existent-model-core", mock_model_config_for_core)
    assert "not found in configured models" in config_result["error"]

def test_get_target_api_config_no_models_configured_core(mock_model_config_empty_for_core):
    config_result = get_target_api_config("any-model-core", mock_model_config_empty_for_core)
    assert "No models configured" in config_result["error"]

def test_get_target_api_config_openai_missing_apibase_core():
    faulty_config = [{"provider": "openai", "model": "openai-error-core", "apiKey": "key"}] # apiBase missing
    config_result = get_target_api_config("openai-error-core", faulty_config)
    assert "apiBase not configured" in config_result["error"]


# --- Tests for handle_anthropic_sdk_request (formerly _handle_anthropic_sdk_request) ---
# We need a Flask app context for creating response objects (jsonify, Response)
# and using stream_with_context.
@pytest.fixture
def app_context():
    app = Flask(__name__)
    with app.app_context():
        yield

@pytest.mark.parametrize("is_stream", [True, False])
@patch('dolphin_logger.core_proxy._should_log_request') # Path to _should_log_request in core_proxy
@patch('dolphin_logger.core_proxy.get_current_log_file')
# log_lock is imported directly, so patching it in core_proxy if it's used there.
# If it's a global in logging_utils, it's already imported.
@patch('builtins.open', new_callable=mock_open)
@patch('dolphin_logger.core_proxy.anthropic.Anthropic') # Path to Anthropic client in core_proxy
def test_handle_anthropic_sdk_request_core(
    mock_anthropic_class, mock_builtin_open, mock_get_log_file, mock_should_log,
    is_stream, app_context # Ensure Flask app context
):
    # Access log_lock from where it's defined if it's used by the handler
    # from dolphin_logger.logging_utils import log_lock # If it were a separate mock target

    mock_should_log.return_value = True # Assume we should log for these tests
    mock_get_log_file.return_value = Path(f"/tmp/test_anthropic_core_{is_stream}.jsonl")
    
    mock_anthropic_instance = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_instance
    
    original_request_data = {"model": "claude-client-core", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}
    # json_data_for_sdk is what's actually passed to the Anthropic client constructor (after model name mapping)
    json_data_for_sdk_call = {"model": "claude-provider-core", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream, "max_tokens": 4096}

    expected_response_content = "Hello from Anthropic (core)!"
    if is_stream:
        mock_stream_chunk_delta1 = MagicMock(type="content_block_delta", delta=MagicMock(text="Hello from "))
        mock_stream_chunk_delta2 = MagicMock(type="content_block_delta", delta=MagicMock(text="Anthropic (core)!"))
        mock_stream_stop_chunk = MagicMock(type="message_stop", message=MagicMock(stop_reason="end_turn"))
        mock_anthropic_instance.messages.create.return_value = [mock_stream_chunk_delta1, mock_stream_chunk_delta2, mock_stream_stop_chunk]
    else:
        mock_sdk_response = MagicMock(
            id="anthropic-id-core", model="claude-provider-core-response", role="assistant",
            content=[MagicMock(text=expected_response_content)], stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=25)
        )
        mock_anthropic_instance.messages.create.return_value = mock_sdk_response

    # Call the handler function from core_proxy
    response_flask = handle_anthropic_sdk_request(
        json_data_for_sdk=json_data_for_sdk_call, target_model="claude-provider-core", 
        target_api_key=MOCK_ANTHROPIC_KEY_CORE, is_stream=is_stream, 
        original_request_json_data=original_request_data
    )

    assert response_flask.status_code == 200
    # Arguments for the Anthropic client call
    anthropic_call_args = {
        "model": "claude-provider-core", 
        "messages": json_data_for_sdk_call["messages"],
        "max_tokens": json_data_for_sdk_call["max_tokens"], # Check default or passed value
        "stream": is_stream
    }
    mock_anthropic_instance.messages.create.assert_called_once_with(**anthropic_call_args)

    if is_stream:
        assert 'text/event-stream' in response_flask.content_type
        # Simplified check for stream content
        stream_content_full = b"".join(response_flask.response).decode()
        assert expected_response_content in stream_content_full
    else:
        response_json = json.loads(response_flask.data.decode()) # jsonify returns Response object
        assert response_json["choices"][0]["message"]["content"] == expected_response_content

    mock_should_log.assert_called_once_with(original_request_data)
    mock_builtin_open.assert_called_once_with(Path(f"/tmp/test_anthropic_core_{is_stream}.jsonl"), 'a')
    logged_data_str = mock_builtin_open().write.call_args[0][0]
    logged_data = json.loads(logged_data_str)
    assert logged_data["request"] == original_request_data
    assert logged_data["response"] == expected_response_content


@patch('dolphin_logger.core_proxy.anthropic.Anthropic')
def test_handle_anthropic_sdk_request_api_error_core(mock_anthropic_class, app_context):
    mock_anthropic_instance = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_instance
    # Import APIError from where it's used (dolphin_logger.core_proxy.anthropic)
    from dolphin_logger.core_proxy import anthropic as anthropic_module_in_core_proxy
    mock_anthropic_instance.messages.create.side_effect = anthropic_module_in_core_proxy.APIError(
        message="Test Anthropic API Error Core", request=MagicMock(), body={"type": "test_error_core"},
        status_code=401 
    )
    response_flask = handle_anthropic_sdk_request({}, "model", MOCK_ANTHROPIC_KEY_CORE, False, {})
    assert response_flask.status_code == 401
    json_data = json.loads(response_flask.data.decode())
    assert json_data["error"]["message"] == "Test Anthropic API Error Core"
    assert json_data["error"]["type"] == "test_error_core"

# --- Tests for handle_rest_api_request (formerly _handle_rest_api_request) ---
@pytest.mark.parametrize("is_stream, provider_name", [(True, "openai"), (False, "openai"), (True, "ollama"), (False, "ollama")])
@patch('dolphin_logger.core_proxy._should_log_request')
@patch('dolphin_logger.core_proxy.get_current_log_file')
@patch('builtins.open', new_callable=mock_open)
@patch('dolphin_logger.core_proxy.requests.request') # Path to requests.request in core_proxy
def test_handle_rest_api_request_core(
    mock_requests_request, mock_builtin_open, mock_get_log_file, mock_should_log,
    is_stream, provider_name, app_context # app_context for Flask responses
):
    mock_should_log.return_value = True
    mock_get_log_file.return_value = Path(f"/tmp/test_rest_core_{provider_name}_{is_stream}.jsonl")

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.headers = {'Content-Type': 'application/json' if not is_stream else 'text/event-stream'}
    
    original_request_data = {"model": f"{provider_name}-client-core", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}
    data_bytes_to_send = json.dumps({"model": f"{provider_name}-provider-core", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}).encode('utf-8')
    
    expected_response_content = f"Hello from {provider_name} REST (core)!"

    if is_stream:
        lines = [f'data: {json.dumps({"choices": [{"delta": {"content": "Hello from "}}]})}\n\n'.encode(),
                 f'data: {json.dumps({"choices": [{"delta": {"content": f"{provider_name} REST (core)!"}}]})}\n\n'.encode(),
                 f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})}\n\n'.encode(),
                 b'data: [DONE]\n\n']
        mock_api_response.iter_lines.return_value = iter(lines)
    else:
        api_response_dict = {"choices": [{"message": {"content": expected_response_content}, "finish_reason": "stop"}]}
        mock_api_response.json.return_value = api_response_dict
        mock_api_response.content = json.dumps(api_response_dict).encode('utf-8')

    mock_requests_request.return_value = mock_api_response

    url = f"http://localhost/{provider_name}/v1/chat/completions"
    headers_to_send = {"Host": "localhost"}
    if provider_name == "openai": headers_to_send["Authorization"] = f"Bearer {MOCK_OPENAI_KEY_CORE}"

    response_flask = handle_rest_api_request(
        "POST", url, headers_to_send, data_bytes_to_send, is_stream, original_request_data
    )

    assert response_flask.status_code == 200
    mock_requests_request.assert_called_once_with(
        method="POST", url=url, headers=headers_to_send, data=data_bytes_to_send, stream=is_stream, timeout=300
    )

    if is_stream:
        assert 'text/event-stream' in response_flask.content_type
        stream_content_full = b"".join(response_flask.response).decode()
        assert expected_response_content in stream_content_full
    else:
        assert response_flask.data == mock_api_response.content # Raw passthrough for non-stream

    mock_should_log.assert_called_once_with(original_request_data)
    mock_builtin_open.assert_called_once_with(Path(f"/tmp/test_rest_core_{provider_name}_{is_stream}.jsonl"), 'a')
    logged_data = json.loads(mock_builtin_open().write.call_args[0][0])
    assert logged_data["request"] == original_request_data
    assert logged_data["response"] == expected_response_content


@patch('dolphin_logger.core_proxy.requests.request')
def test_handle_rest_api_request_connection_error_core(mock_requests_request, app_context):
    # Import RequestException from where it's used
    from dolphin_logger.core_proxy import requests as requests_in_core_proxy 
    mock_requests_request.side_effect = requests_in_core_proxy.exceptions.RequestException("Connection failed core")
    
    response_flask = handle_rest_api_request("POST", "url_core", {}, b'{}', False, {})
    assert response_flask.status_code == 502
    json_data = json.loads(response_flask.data.decode())
    assert "Error connecting to upstream API: Connection failed core" in json_data["error"]["message"]

@patch('dolphin_logger.core_proxy.requests.request')
def test_handle_rest_api_http_error_core(mock_requests_request, app_context):
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 403 # Forbidden
    mock_upstream_response.headers = {'Content-Type': 'application/json'}
    mock_upstream_response.json.return_value = {"error": {"message": "Upstream auth error core"}}
    
    from dolphin_logger.core_proxy import requests as requests_in_core_proxy
    mock_upstream_response.raise_for_status = MagicMock(
        side_effect=requests_in_core_proxy.exceptions.HTTPError(response=mock_upstream_response)
    )
    mock_requests_request.return_value = mock_upstream_response
    
    response_flask = handle_rest_api_request("POST", "url_core_http_error", {}, b'{}', False, {})
    assert response_flask.status_code == 403
    json_data = json.loads(response_flask.data.decode())
    assert "Upstream auth error core" in json_data["error"]["message"]

```
