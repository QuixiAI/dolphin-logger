import pytest
from unittest.mock import patch, mock_open, MagicMock, call
import os
import json
from pathlib import Path
from datetime import date, datetime, timedelta
import uuid

# Ensure the test can find the module (adjust if your structure differs)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dolphin_logger import main as dolphin_main_module
from dolphin_logger.main import app as flask_app
from dolphin_logger.main import _should_log_request, _get_target_api_config, get_current_log_file, _handle_anthropic_sdk_request, _handle_rest_api_request

# Common test data
MOCK_ANTHROPIC_KEY = "fake_anthropic_key_123"
MOCK_OPENAI_KEY = "fake_openai_key_123"
MOCK_ENV_KEY_NAME = "MY_MODEL_API_KEY"
MOCK_ENV_KEY_VALUE = "resolved_env_key_456"

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset globals before each test to ensure test isolation."""
    dolphin_main_module.current_logfile_name = None
    dolphin_main_module.current_logfile_date = None
    # It's crucial to reset MODEL_CONFIG as it's manipulated in many tests
    # and used by the app context.
    dolphin_main_module.MODEL_CONFIG = []


@pytest.fixture
def client():
    """A test client for the Flask app."""
    with flask_app.test_client() as client:
        with flask_app.app_context():
             # Ensure MODEL_CONFIG is clean before client is used for a test run
            dolphin_main_module.MODEL_CONFIG = []
            yield client

@pytest.fixture
def mock_model_config_standard():
    return [
        {
            "provider": "openai", "providerModel": "gpt-4", "model": "openai-gpt4",
            "apiBase": "https://api.openai.com/v1", "apiKey": MOCK_OPENAI_KEY
        },
        {
            "provider": "anthropic", "providerModel": "claude-3-opus-20240229", "model": "anthropic-claude3",
            "apiKey": MOCK_ANTHROPIC_KEY
        },
        {
            "provider": "ollama", "providerModel": "llama3", "model": "ollama-llama3",
            "apiBase": "http://localhost:11434/v1"
        },
        {
            "provider": "openai", "providerModel": "gpt-3.5-turbo", "model": "openai-gpt3.5-env",
            "apiBase": "https://api.openai.com/v1", "apiKey": f"ENV:{MOCK_ENV_KEY_NAME}"
        }
    ]

@pytest.fixture
def mock_model_config_empty():
    return []

# --- Tests for Health Check ---
def test_health_check_with_config(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {"status": "ok", "message": "Server is healthy, configuration loaded."}

def test_health_check_without_config(client, mock_model_config_empty):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_empty
    response = client.get('/health')
    assert response.status_code == 500
    json_data = response.get_json()
    assert json_data == {"status": "error", "message": "Server is running, but configuration might have issues (e.g., no models loaded)."}

# --- Tests for /v1/models Endpoint ---
def test_models_endpoint_with_config(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.get('/v1/models')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["object"] == "list"
    assert len(json_data["data"]) == len(mock_model_config_standard)
    for i, model_conf in enumerate(mock_model_config_standard):
        assert json_data["data"][i]["id"] == model_conf["model"]
        assert json_data["data"][i]["owned_by"] == model_conf["provider"]
        assert json_data["data"][i]["provider_model"] == model_conf.get("providerModel", "")

def test_models_endpoint_empty_config(client, mock_model_config_empty):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_empty
    response = client.get('/v1/models')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {"data": [], "object": "list"}

# --- Tests for _get_target_api_config ---
@pytest.mark.parametrize("requested_model_id, expected_provider, expected_api_url_contains, expected_key_or_resolved, expected_target_model", [
    ("openai-gpt4", "openai", "api.openai.com/v1", MOCK_OPENAI_KEY, "gpt-4"),
    ("anthropic-claude3", "anthropic", "anthropic_sdk", MOCK_ANTHROPIC_KEY, "claude-3-opus-20240229"),
    ("ollama-llama3", "ollama", "localhost:11434/v1", "", "llama3"),
    ("openai-gpt3.5-env", "openai", "api.openai.com/v1", MOCK_ENV_KEY_VALUE, "gpt-3.5-turbo"),
])
def test_get_target_api_config_found(mock_model_config_standard, requested_model_id, expected_provider, expected_api_url_contains, expected_key_or_resolved, expected_target_model, mocker):
    if requested_model_id == "openai-gpt3.5-env":
        mocker.patch.dict(os.environ, {MOCK_ENV_KEY_NAME: MOCK_ENV_KEY_VALUE})
        # Simulate that load_config already processed this.
        for m_cfg in mock_model_config_standard:
            if m_cfg["model"] == "openai-gpt3.5-env":
                m_cfg["apiKey"] = MOCK_ENV_KEY_VALUE # Pre-resolve for the test
    
    config_result = _get_target_api_config(requested_model_id, {}, mock_model_config_standard)
    
    assert config_result["error"] is None
    assert config_result["provider"] == expected_provider
    assert expected_api_url_contains in config_result["target_api_url"]
    assert config_result["target_api_key"] == expected_key_or_resolved
    assert config_result["target_model"] == expected_target_model

def test_get_target_api_config_not_found(mock_model_config_standard):
    config_result = _get_target_api_config("non-existent-model", {}, mock_model_config_standard)
    assert "not found in configured models" in config_result["error"]

def test_get_target_api_config_no_models_configured(mock_model_config_empty):
    config_result = _get_target_api_config("any-model", {}, mock_model_config_empty)
    assert "No models configured" in config_result["error"]

def test_get_target_api_config_openai_missing_apibase():
    faulty_config = [{"provider": "openai", "model": "openai-error", "apiKey": "key"}] # apiBase missing
    config_result = _get_target_api_config("openai-error", {}, faulty_config)
    assert "apiBase not configured" in config_result["error"]

# --- Test for main proxy() route error handling ---
def test_proxy_model_not_found_error_in_proxy_route(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.post('/v1/chat/completions', json={
        "model": "model-does-not-exist", "messages": [{"role": "user", "content": "Hello"}]
    })
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Model 'model-does-not-exist' not found" in json_data["error"]

# --- Tests for _handle_anthropic_sdk_request ---
COMMON_PATCHES_LOGGING_ANTHROPIC = [
    patch(f'{dolphin_main_module.__name__}._should_log_request'),
    patch(f'{dolphin_main_module.__name__}.get_current_log_file'),
    patch(f'{dolphin_main_module.__name__}.log_lock'), # Assuming log_lock is used correctly
    patch('builtins.open', new_callable=mock_open),
    patch(f'{dolphin_main_module.__name__}.anthropic.Anthropic')
]

@pytest.mark.parametrize("is_stream", [True, False])
@patch(f'{dolphin_main_module.__name__}._should_log_request')
@patch(f'{dolphin_main_module.__name__}.get_current_log_file')
@patch(f'{dolphin_main_module.__name__}.log_lock')
@patch('builtins.open', new_callable=mock_open)
@patch(f'{dolphin_main_module.__name__}.anthropic.Anthropic')
def test_handle_anthropic_sdk_request_streaming_and_non_streaming(
    mock_anthropic_class, mock_builtin_open, mock_log_lock, mock_get_log_file, mock_should_log,
    is_stream, client # client for app context
):
    mock_should_log.return_value = True
    mock_get_log_file.return_value = Path(f"/tmp/test_anthropic_{is_stream}.jsonl")
    
    mock_anthropic_instance = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_instance
    
    original_request_data = {"model": "claude-client", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}
    json_data_for_sdk = {"model": "claude-provider", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream, "max_tokens": 100} # providerModel and defaults added

    if is_stream:
        # Mock streaming response
        mock_stream_chunk_delta = MagicMock()
        mock_stream_chunk_delta.delta.text = "Hello "
        mock_stream_chunk_delta.type = "content_block_delta"
        
        mock_stream_chunk_delta2 = MagicMock()
        mock_stream_chunk_delta2.delta.text = "Anthropic!"
        mock_stream_chunk_delta2.type = "content_block_delta"

        mock_stream_stop_chunk = MagicMock(type="message_stop")
        mock_stream_stop_chunk.message = MagicMock(stop_reason="end_turn")

        mock_anthropic_instance.messages.create.return_value = [mock_stream_chunk_delta, mock_stream_chunk_delta2, mock_stream_stop_chunk]
        expected_response_content = "Hello Anthropic!"
    else:
        # Mock non-streaming response
        mock_sdk_response = MagicMock(
            id="anthropic-id", model="claude-provider-response", role="assistant",
            content=[MagicMock(text="Hello Anthropic!")], stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=20)
        )
        mock_anthropic_instance.messages.create.return_value = mock_sdk_response
        expected_response_content = "Hello Anthropic!"

    response_flask = _handle_anthropic_sdk_request(
        json_data_for_sdk=json_data_for_sdk, target_model="claude-provider", 
        target_api_key=MOCK_ANTHROPIC_KEY, is_stream=is_stream, original_request_json_data=original_request_data
    )

    assert response_flask.status_code == 200
    mock_anthropic_instance.messages.create.assert_called_once_with(
        model="claude-provider", messages=json_data_for_sdk["messages"], 
        max_tokens=json_data_for_sdk["max_tokens"], stream=is_stream
    )

    if is_stream:
        assert response_flask.content_type == 'text/event-stream'
        # Collect stream content
        stream_content_parts = [json.loads(line.decode().split("data: ")[1]) for line in response_flask.response if line.startswith(b"data: ") and line != b"data: [DONE]\n\n"]
        actual_text = "".join(c["choices"][0]["delta"].get("content","") for c in stream_content_parts if "delta" in c["choices"][0] and "content" in c["choices"][0]["delta"])
        assert actual_text == expected_response_content
        assert stream_content_parts[-1]["choices"][0]["finish_reason"] == "end_turn"
    else:
        response_json = json.loads(response_flask.data)
        assert response_json["choices"][0]["message"]["content"] == expected_response_content
        assert response_json["choices"][0]["finish_reason"] == "end_turn"

    mock_should_log.assert_called_once_with(original_request_data)
    mock_builtin_open.assert_called_once_with(Path(f"/tmp/test_anthropic_{is_stream}.jsonl"), 'a')
    logged_data = json.loads(mock_builtin_open().write.call_args[0][0])
    assert logged_data["request"] == original_request_data
    assert logged_data["response"] == expected_response_content

@patch(f'{dolphin_main_module.__name__}.anthropic.Anthropic')
def test_handle_anthropic_sdk_request_api_error(mock_anthropic_class, client):
    mock_anthropic_instance = MagicMock()
    mock_anthropic_class.return_value = mock_anthropic_instance
    mock_anthropic_instance.messages.create.side_effect = dolphin_main_module.anthropic.APIError(
        message="Test Anthropic API Error", request=MagicMock(), body={"type": "test_error"},
        status_code=400 
    )
    response_flask = _handle_anthropic_sdk_request({}, "model", MOCK_ANTHROPIC_KEY, False, {})
    assert response_flask.status_code == 400
    json_data = json.loads(response_flask.data)
    assert json_data["error"]["message"] == "Test Anthropic API Error"
    assert json_data["error"]["type"] == "test_error"

# --- Tests for _handle_rest_api_request ---
@pytest.mark.parametrize("is_stream, provider", [(True, "openai"), (False, "openai"), (True, "ollama"), (False, "ollama")])
@patch(f'{dolphin_main_module.__name__}._should_log_request')
@patch(f'{dolphin_main_module.__name__}.get_current_log_file')
@patch(f'{dolphin_main_module.__name__}.log_lock')
@patch('builtins.open', new_callable=mock_open)
@patch(f'{dolphin_main_module.__name__}.requests.request')
def test_handle_rest_api_request_streaming_and_non_streaming(
    mock_requests_request, mock_builtin_open, mock_log_lock, mock_get_log_file, mock_should_log,
    is_stream, provider, client # client for app context
):
    mock_should_log.return_value = True
    mock_get_log_file.return_value = Path(f"/tmp/test_rest_{provider}_{is_stream}.jsonl")

    mock_api_response = MagicMock()
    mock_api_response.status_code = 200
    mock_api_response.headers = {'Content-Type': 'application/json' if not is_stream else 'text/event-stream'}

    original_request_data = {"model": f"{provider}-client", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}
    data_bytes_to_send = json.dumps({"model": f"{provider}-provider", "messages": [{"role": "user", "content": "Hi"}], "stream": is_stream}).encode('utf-8')
    
    expected_response_content = f"Hello from {provider} REST!"

    if is_stream:
        lines = [
            b'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": f"from {provider} "}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "REST!"}}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        mock_api_response.iter_lines.return_value = iter(lines)
        mock_api_response.headers['Content-Type'] = 'text/event-stream'
    else:
        api_response_content = {
            "choices": [{"message": {"content": expected_response_content}, "finish_reason": "stop"}]
        }
        mock_api_response.json.return_value = api_response_content
        mock_api_response.content = json.dumps(api_response_content).encode('utf-8')

    mock_requests_request.return_value = mock_api_response

    url = f"http://localhost/{provider}/v1/chat/completions"
    headers = {"Host": "localhost"}
    if provider == "openai": headers["Authorization"] = f"Bearer {MOCK_OPENAI_KEY}"

    response_flask = _handle_rest_api_request(
        "POST", url, headers, data_bytes_to_send, is_stream, original_request_data
    )

    assert response_flask.status_code == 200
    mock_requests_request.assert_called_once_with(
        method="POST", url=url, headers=headers, data=data_bytes_to_send, stream=is_stream, timeout=300
    )

    if is_stream:
        assert response_flask.content_type == 'text/event-stream'
        # Iterating through response.response which is the generator
        stream_data_bytes = b"".join(response_flask.response)
        # Reconstruct text from stream for comparison
        actual_text_parts = []
        for line_bytes in stream_data_bytes.split(b'\n\n'):
            if line_bytes.startswith(b'data: '):
                data_part = line_bytes.replace(b'data: ', b'')
                if data_part == b'[DONE]': break
                try:
                    chunk_json = json.loads(data_part.decode())
                    if chunk_json["choices"][0].get("delta", {}).get("content"):
                        actual_text_parts.append(chunk_json["choices"][0]["delta"]["content"])
                except: pass # Ignore non-json or non-delta containing lines for text reconstruction
        assert "".join(actual_text_parts) == expected_response_content
    else:
        assert response_flask.data == mock_api_response.content # Raw passthrough

    mock_should_log.assert_called_once_with(original_request_data)
    mock_builtin_open.assert_called_once_with(Path(f"/tmp/test_rest_{provider}_{is_stream}.jsonl"), 'a')
    logged_data = json.loads(mock_builtin_open().write.call_args[0][0])
    assert logged_data["request"] == original_request_data
    assert logged_data["response"] == expected_response_content


@patch(f'{dolphin_main_module.__name__}.requests.request')
def test_handle_rest_api_request_connection_error(mock_requests_request, client):
    mock_requests_request.side_effect = dolphin_main_module.requests.exceptions.RequestException("Connection failed")
    response_flask = _handle_rest_api_request("POST", "url", {}, b'{}', False, {})
    assert response_flask.status_code == 502 # Bad Gateway for connection issues
    json_data = json.loads(response_flask.data)
    assert "Error connecting to upstream API: Connection failed" in json_data["error"]["message"]

# --- Tests for _should_log_request ---
@pytest.mark.parametrize("request_data, expected_log_decision", [
    ({"messages": [{"role": "user", "content": "Log this"}]}, True),
    ({"messages": [{"role": "user", "content": "### Task: Do not log this"}]}, False),
])
def test_should_log_request_various_scenarios(request_data, expected_log_decision, mocker):
    mocker.patch('builtins.print') # Suppress print from the function
    assert _should_log_request(request_data) == expected_log_decision

# --- Tests for get_current_log_file ---
@patch(f'{dolphin_main_module.__name__}.uuid.uuid4')
@patch(f'{dolphin_main_module.__name__}.date')
@patch(f'{dolphin_main_module.__name__}.os.listdir')
@patch(f'{dolphin_main_module.__name__}.os.path.getmtime')
@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
def test_get_current_log_file_creation_and_resume_logic(
    mock_get_logs_dir, mock_getmtime, mock_listdir, mock_date, mock_uuid, tmp_path
):
    # Setup common mock returns
    mock_logs_dir_path = tmp_path / "test_logs_main"
    mock_logs_dir_path.mkdir(exist_ok=True)
    mock_get_logs_dir.return_value = mock_logs_dir_path
    
    today_date = date(2023, 10, 5)
    mock_date.today.return_value = today_date
    
    # --- Test 1: No existing files, creates new ---
    dolphin_main_module.current_logfile_name = None # Reset state
    dolphin_main_module.current_logfile_date = None
    
    new_uuid_val = uuid.UUID("11111111-1111-1111-1111-111111111111")
    mock_uuid.return_value = new_uuid_val
    mock_listdir.return_value = []
    
    log_file = get_current_log_file()
    expected_new_path = mock_logs_dir_path / f"{new_uuid_val}.jsonl"
    assert log_file == expected_new_path
    assert dolphin_main_module.current_logfile_name == expected_new_path
    assert dolphin_main_module.current_logfile_date == today_date
    mock_uuid.assert_called_once() # Called once to create the new file name

    # --- Test 2: Existing file from today, should resume (latest one) ---
    dolphin_main_module.current_logfile_name = None # Reset state
    dolphin_main_module.current_logfile_date = None
    mock_uuid.reset_mock() # Reset call count for uuid

    exist_uuid1_str = "22222222-2222-2222-2222-222222222222"
    exist_uuid2_str = "33333333-3333-3333-3333-333333333333" # This one will be "later"
    mock_listdir.return_value = [f"{exist_uuid1_str}.jsonl", f"{exist_uuid2_str}.jsonl", "not-a-uuid.txt"]
    
    # Timestamps for today
    time_early = datetime(today_date.year, today_date.month, today_date.day, 9, 0, 0).timestamp()
    time_later = datetime(today_date.year, today_date.month, today_date.day, 10, 0, 0).timestamp()

    def getmtime_side_effect_resume(p):
        if str(p).endswith(f"{exist_uuid1_str}.jsonl"): return time_early
        if str(p).endswith(f"{exist_uuid2_str}.jsonl"): return time_later
        return 0.0
    mock_getmtime.side_effect = getmtime_side_effect_resume
    
    log_file = get_current_log_file()
    expected_resume_path = mock_logs_dir_path / f"{exist_uuid2_str}.jsonl" # Should pick the later one
    assert log_file == expected_resume_path
    assert dolphin_main_module.current_logfile_name == expected_resume_path
    assert dolphin_main_module.current_logfile_date == today_date
    mock_uuid.assert_not_called() # Should not create a new UUID

    # --- Test 3: Date changes, should create new file ---
    # Globals are already set from previous test (log file from 2023-10-05)
    # current_logfile_name = expected_resume_path
    # current_logfile_date = today_date
    
    new_day_date = date(2023, 10, 6) # Next day
    mock_date.today.return_value = new_day_date
    
    new_day_uuid_val = uuid.UUID("44444444-4444-4444-4444-444444444444")
    mock_uuid.return_value = new_day_uuid_val # Reset mock for a new UUID
    mock_listdir.return_value = [] # No files for the new day yet

    log_file = get_current_log_file()
    expected_new_day_path = mock_logs_dir_path / f"{new_day_uuid_val}.jsonl"
    assert log_file == expected_new_day_path
    assert dolphin_main_module.current_logfile_name == expected_new_day_path
    assert dolphin_main_module.current_logfile_date == new_day_date
    mock_uuid.assert_called_once() # Called once for the new day's file


@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
@patch(f'{dolphin_main_module.__name__}.date')
@patch(f'{dolphin_main_module.__name__}.os.listdir')
@patch(f'{dolphin_main_module.__name__}.os.path.getmtime', side_effect=OSError("Permission denied"))
def test_get_current_log_file_handles_oserror_getmtime(
    mock_getmtime_oserror, mock_listdir, mock_date, mock_get_logs_dir, tmp_path, mocker
):
    # Reset globals
    dolphin_main_module.current_logfile_name = None
    dolphin_main_module.current_logfile_date = None

    mock_logs_dir_path = tmp_path / "logs_oserror"
    mock_logs_dir_path.mkdir()
    mock_get_logs_dir.return_value = mock_logs_dir_path

    mock_date.today.return_value = date(2023, 1, 1)
    # Simulate a file that will cause OSError on getmtime
    mock_listdir.return_value = ["problematic_file.jsonl"] 
    
    # Mock uuid.uuid4 for new file creation
    new_uuid = uuid.uuid4()
    mocker.patch(f'{dolphin_main_module.__name__}.uuid.uuid4', return_value=new_uuid)

    log_file = get_current_log_file()
    
    # Should skip the problematic file and create a new one
    expected_path = mock_logs_dir_path / f"{new_uuid}.jsonl"
    assert log_file == expected_path
    # getmtime was called for problematic_file.jsonl, then it should create a new one.
    mock_getmtime_oserror.assert_called_once_with(mock_logs_dir_path / "problematic_file.jsonl")
    # Ensure print was called with a warning or info about skipping
    # This part depends on if you add print statements for such errors.
    # For now, just ensure it falls back to new file creation.
    assert dolphin_main_module.current_logfile_name == expected_path
    assert dolphin_main_module.current_logfile_date == date(2023, 1, 1)

# Test options handler
def test_options_handler(client):
    response = client.options('/v1/chat/completions')
    assert response.status_code == 200 # Default Flask options response is 200
    # Check for expected CORS headers if explicitly set by `app.make_default_options_response()`
    # or if CORS extension handles it. For now, just status is fine.

# Test proxy with invalid JSON
def test_proxy_invalid_json(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.post('/v1/chat/completions', data="this is not json {", content_type='application/json')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid JSON" in json_data["error"]

# Test proxy with non-POST request (that isn't /models or /health)
def test_proxy_non_post_request_error(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.get('/v1/chat/completions') # GET instead of POST
    assert response.status_code == 400 # Or 405 Method Not Allowed, depending on strictness. Current code gives 400.
    json_data = response.get_json()
    assert "Proxying requires a POST request" in json_data["error"]

# Test proxy with POST but no 'model' field
def test_proxy_post_no_model_field(client, mock_model_config_standard):
    dolphin_main_module.MODEL_CONFIG = mock_model_config_standard
    response = client.post('/v1/chat/completions', json={"messages": [{"role": "user", "content": "Hi"}]})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "must include a 'model' field" in json_data["error"]

# Test _handle_anthropic_sdk_request missing API key
@patch(f'{dolphin_main_module.__name__}.anthropic.Anthropic')
def test_handle_anthropic_sdk_missing_key(mock_anthropic_class, client):
    response = _handle_anthropic_sdk_request({}, "model", None, False, {}) # None for API key
    assert response.status_code == 500
    json_data = json.loads(response.data)
    assert "Anthropic API key is missing" in json_data["error"]["message"]

# Test _handle_rest_api_request HTTP error from upstream
@patch(f'{dolphin_main_module.__name__}.requests.request')
def test_handle_rest_api_http_error(mock_requests_request, client):
    mock_upstream_response = MagicMock()
    mock_upstream_response.status_code = 401 # Unauthorized
    mock_upstream_response.headers = {'Content-Type': 'application/json'}
    mock_upstream_response.json.return_value = {"error": {"message": "Upstream auth error"}}
    # Need to make raise_for_status work correctly for HTTPError
    mock_upstream_response.raise_for_status = MagicMock(side_effect=dolphin_main_module.requests.exceptions.HTTPError(response=mock_upstream_response))
    
    mock_requests_request.return_value = mock_upstream_response
    
    response = _handle_rest_api_request("POST", "url", {}, b'{}', False, {})
    assert response.status_code == 401
    json_data = json.loads(response.data)
    assert "Upstream auth error" in json_data["error"]["message"]

# Test run_server and cli (basic smoke tests, not full server lifecycle)
@patch(f'{dolphin_main_module.__name__}.load_config')
@patch(f'{dolphin_main_module.__name__}.app.run')
def test_run_server(mock_app_run, mock_load_config, client):
    mock_load_config.return_value = {"models": [{"id": "test"}]}
    dolphin_main_module.run_server()
    mock_load_config.assert_called_once()
    mock_app_run.assert_called_once() # Check if Flask's run is called

@patch(f'{dolphin_main_module.__name__}.argparse.ArgumentParser')
@patch(f'{dolphin_main_module.__name__}.load_config')
@patch(f'{dolphin_main_module.__name__}.run_server')
@patch(f'{dolphin_main_module.__name__}.upload_logs')
def test_cli_server_mode(mock_upload_logs, mock_run_server, mock_load_config, mock_argparse, client):
    mock_args = MagicMock(upload=False)
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_args.return_value = mock_args
    mock_argparse.return_value = mock_parser_instance

    dolphin_main_module.cli()
    
    mock_load_config.assert_called_once()
    mock_run_server.assert_called_once()
    mock_upload_logs.assert_not_called()

@patch(f'{dolphin_main_module.__name__}.argparse.ArgumentParser')
@patch(f'{dolphin_main_module.__name__}.load_config')
@patch(f'{dolphin_main_module.__name__}.run_server')
@patch(f'{dolphin_main_module.__name__}.upload_logs')
def test_cli_upload_mode(mock_upload_logs, mock_run_server, mock_load_config, mock_argparse, client):
    mock_args = MagicMock(upload=True)
    mock_parser_instance = MagicMock()
    mock_parser_instance.parse_args.return_value = mock_args
    mock_argparse.return_value = mock_parser_instance

    dolphin_main_module.cli()

    mock_load_config.assert_called_once()
    mock_upload_logs.assert_called_once()
    mock_run_server.assert_not_called()

# Test for find_jsonl_files (from upload logic, but in main.py)
@patch(f'{dolphin_main_module.__name__}.glob.glob')
@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
def test_find_jsonl_files(mock_get_logs_dir, mock_glob_glob):
    mock_logs_dir_path = Path("/mock/logs")
    mock_get_logs_dir.return_value = mock_logs_dir_path
    expected_files = ["log1.jsonl", "log2.jsonl"]
    mock_glob_glob.return_value = expected_files

    result = dolphin_main_module.find_jsonl_files(mock_logs_dir_path)
    assert result == expected_files
    mock_glob_glob.assert_called_once_with(os.path.join(mock_logs_dir_path, "*.jsonl"))

# Test for upload_logs (more of an integration test, mock HfApi heavily)
@patch(f'{dolphin_main_module.__name__}.HfApi')
@patch(f'{dolphin_main_module.__name__}.find_jsonl_files')
@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
def test_upload_logs_success(mock_get_logs_dir, mock_find_jsonl, mock_hf_api_class, client):
    mock_logs_dir_path = Path("/mock/logs_upload")
    mock_get_logs_dir.return_value = mock_logs_dir_path
    mock_find_jsonl.return_value = ["/mock/logs_upload/log1.jsonl", "/mock/logs_upload/log2.jsonl"]

    mock_hf_api_instance = MagicMock()
    mock_hf_api_class.return_value = mock_hf_api_instance
    mock_hf_api_instance.repo_info.return_value = MagicMock() # Simulate repo exists
    
    commit_info_mock = MagicMock()
    commit_info_mock.pr_url = "http://hf.co/pr/1"
    mock_hf_api_instance.create_commit.return_value = commit_info_mock

    with patch('builtins.print') as mock_print: # Suppress prints
        dolphin_main_module.upload_logs()

    mock_hf_api_instance.create_branch.assert_called_once() # Default branch name pattern check is tricky
    assert mock_hf_api_instance.create_commit.call_count == 1
    args, kwargs = mock_hf_api_instance.create_commit.call_args
    assert kwargs['repo_id'] == dolphin_main_module.DATASET_REPO_ID
    assert kwargs['create_pr'] == True
    assert len(kwargs['operations']) == 2 # Two files
    assert kwargs['operations'][0].path_in_repo == "log1.jsonl"
    assert kwargs['operations'][1].path_in_repo == "log2.jsonl"
    
    # Check if PR URL was printed
    printed_pr_url = False
    for call_arg in mock_print.call_args_list:
        if "Pull Request (Draft): http://hf.co/pr/1" in call_arg[0][0]:
            printed_pr_url = True
            break
    assert printed_pr_url

@patch(f'{dolphin_main_module.__name__}.HfApi')
@patch(f'{dolphin_main_module.__name__}.find_jsonl_files')
@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
def test_upload_logs_no_files(mock_get_logs_dir, mock_find_jsonl, mock_hf_api_class, client):
    mock_get_logs_dir.return_value = Path("/mock/logs_empty")
    mock_find_jsonl.return_value = [] # No files found

    with patch('builtins.print') as mock_print:
        dolphin_main_module.upload_logs()
    
    mock_hf_api_class.assert_not_called() # HfApi should not even be instantiated if no files
    
    no_files_printed = False
    for call_arg in mock_print.call_args_list:
        if "No .jsonl files found" in call_arg[0][0]:
            no_files_printed = True
            break
    assert no_files_printed

# Test for upload_logs when branch creation fails
@patch(f'{dolphin_main_module.__name__}.HfApi')
@patch(f'{dolphin_main_module.__name__}.find_jsonl_files')
@patch(f'{dolphin_main_module.__name__}.get_logs_dir')
def test_upload_logs_branch_creation_fails(mock_get_logs_dir, mock_find_jsonl, mock_hf_api_class, client):
    mock_get_logs_dir.return_value = Path("/mock/logs_branch_fail")
    mock_find_jsonl.return_value = ["/mock/logs_branch_fail/log.jsonl"]

    mock_hf_api_instance = MagicMock()
    mock_hf_api_class.return_value = mock_hf_api_instance
    mock_hf_api_instance.repo_info.return_value = MagicMock()
    mock_hf_api_instance.create_branch.side_effect = Exception("Branch creation failed")

    with patch('builtins.print') as mock_print:
        dolphin_main_module.upload_logs()

    mock_hf_api_instance.create_commit.assert_not_called()
    
    failed_branch_printed = False
    for call_arg in mock_print.call_args_list:
        if "Failed to create branch" in call_arg[0][0] and "Aborting PR creation" in call_arg[0][0]:
            failed_branch_printed = True
            break
    assert failed_branch_printedThe tests for `test_main.py` have been written and include coverage for:
- Health Check endpoint (`/health`)
- Models listing endpoint (`/v1/models`)
- `_get_target_api_config` helper for various scenarios (OpenAI, Anthropic, Ollama, ENV var, model not found, missing apiBase).
- `proxy` route error handling for model not found.
- `_handle_anthropic_sdk_request` for streaming and non-streaming, including basic error handling (APIError).
- `_handle_rest_api_request` for streaming and non-streaming (OpenAI & Ollama providers), including basic error handling (RequestException, HTTPError).
- `_should_log_request` for different message contents.
- `get_current_log_file` for new file creation, resuming existing files, and handling date changes. Also includes a test for OSError during `getmtime`.
- Basic CLI argument parsing (`run_server` vs `upload_logs`).
- Basic `upload_logs` functionality (success, no files, branch creation failure).
- Basic `find_jsonl_files`.
- Flask client setup with `app_context`.
- Resetting of relevant globals (`current_logfile_name`, `current_logfile_date`, `MODEL_CONFIG`) using `autouse=True` fixture to ensure test isolation.
- Additional error cases for the main proxy: invalid JSON, non-POST requests, POST without 'model' field.
- Error case for `_handle_anthropic_sdk_request` when API key is missing.

This completes the subtask.
