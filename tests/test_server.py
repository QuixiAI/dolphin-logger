import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the Flask app and other necessary components from the new server module
from dolphin_logger.server import app as flask_app
from dolphin_logger import server as dolphin_server_module # To access/mock MODEL_CONFIG

# Common test data (can be refactored later if shared across many test files)
MOCK_ANTHROPIC_KEY = "fake_anthropic_key_server_test"
MOCK_OPENAI_KEY = "fake_openai_key_server_test"
MOCK_ENV_KEY_NAME_SERVER = "MY_SERVER_TEST_KEY"
MOCK_ENV_KEY_VALUE_SERVER = "resolved_server_env_key"

@pytest.fixture(autouse=True)
def reset_server_globals():
    """Reset globals in server.py before each test."""
    # Primarily MODEL_CONFIG needs reset as it's loaded at server startup
    # and can be manipulated by tests using the client.
    if hasattr(dolphin_server_module, 'MODEL_CONFIG'):
        dolphin_server_module.MODEL_CONFIG = []

@pytest.fixture
def client():
    """A test client for the Flask app."""
    # Ensure MODEL_CONFIG is clean before client is created for a test run
    if hasattr(dolphin_server_module, 'MODEL_CONFIG'):
        dolphin_server_module.MODEL_CONFIG = []
    
    with flask_app.test_client() as client:
        with flask_app.app_context(): # Important for tests that might use app context
            yield client

@pytest.fixture
def mock_model_config_for_server():
    # This config is used to populate dolphin_server_module.MODEL_CONFIG
    return [
        {
            "provider": "openai", "providerModel": "gpt-4-server", "model": "openai-gpt4-server",
            "apiBase": "https://api.openai.com/v1", "apiKey": MOCK_OPENAI_KEY
        },
        {
            "provider": "anthropic", "providerModel": "claude-3-opus-server", "model": "anthropic-claude3-server",
            "apiKey": MOCK_ANTHROPIC_KEY
        },
        {
            "provider": "ollama", "providerModel": "llama3-server", "model": "ollama-llama3-server",
            "apiBase": "http://localhost:11534/v1" # Different port to avoid conflict
        },
        {
            "provider": "openai", "providerModel": "gpt-3.5-turbo-server-env", "model": "openai-gpt3.5-server-env",
            "apiBase": "https://api.openai.com/v1", "apiKey": f"ENV:{MOCK_ENV_KEY_NAME_SERVER}"
        }
    ]

@pytest.fixture
def mock_model_config_empty_for_server():
    return []

# --- Tests for Health Check ---
def test_health_check_with_config(client, mock_model_config_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server
    response = client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {"status": "ok", "message": "Server is healthy, configuration loaded."}

def test_health_check_without_config(client, mock_model_config_empty_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_empty_for_server
    response = client.get('/health')
    assert response.status_code == 500 # As per current health_check logic
    json_data = response.get_json()
    assert "configuration might have issues" in json_data["message"]

# --- Tests for /v1/models Endpoint ---
def test_models_endpoint_with_config(client, mock_model_config_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server
    response = client.get('/v1/models')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["object"] == "list"
    assert len(json_data["data"]) == len(mock_model_config_for_server)
    for i, model_conf in enumerate(mock_model_config_for_server):
        assert json_data["data"][i]["id"] == model_conf["model"]
        assert json_data["data"][i]["owned_by"] == model_conf["provider"]
        assert json_data["data"][i]["provider_model"] == model_conf.get("providerModel", "")

def test_models_endpoint_empty_config(client, mock_model_config_empty_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_empty_for_server
    response = client.get('/v1/models')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {"data": [], "object": "list"}

# --- Tests for main proxy() route error handling (before delegation to core_proxy) ---
def test_proxy_route_invalid_json(client, mock_model_config_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server # Make sure server has some config
    response = client.post('/v1/chat/completions', data="this is not json {", content_type='application/json')
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Invalid JSON" in json_data["error"]

def test_proxy_route_non_post_request_error(client):
    # No MODEL_CONFIG needed as this check is before model lookup
    response = client.get('/v1/chat/completions') 
    assert response.status_code == 400
    json_data = response.get_json()
    assert "primarily proxies POST requests" in json_data["error"]

def test_proxy_route_post_no_model_field(client):
    response = client.post('/v1/chat/completions', json={"messages": [{"role": "user", "content": "Hi"}]})
    assert response.status_code == 400
    json_data = response.get_json()
    assert "must include a 'model' field" in json_data["error"]

def test_proxy_route_model_not_found_in_config(client, mock_model_config_for_server):
    # This tests the part of proxy() that calls get_target_api_config and handles its error return.
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server
    
    # Mock get_target_api_config because we are testing server.py's proxy() handling of its return,
    # not get_target_api_config itself (that's for test_core_proxy.py)
    with patch('dolphin_logger.server.get_target_api_config') as mock_get_config:
        mock_get_config.return_value = {"error": "Model 'model-does-not-exist' not found in configured models."}
        
        response = client.post('/v1/chat/completions', json={
            "model": "model-does-not-exist", 
            "messages": [{"role": "user", "content": "Hello"}]
        })
    assert response.status_code == 400
    json_data = response.get_json()
    assert "Model 'model-does-not-exist' not found" in json_data["error"]

# Test OPTIONS handler (basic check)
def test_options_handler(client):
    response = client.options('/v1/chat/completions')
    assert response.status_code == 200 
    # Further checks for CORS headers could be added if Flask-CORS doesn't handle them sufficiently by default

# Test run_server_main function (mocking app.run and load_config)
@patch('dolphin_logger.server.load_config')
@patch('dolphin_logger.server.app.run')
def test_run_server_main(mock_app_run, mock_load_config):
    # Simulate load_config returning some models
    mock_load_config.return_value = {"models": [{"id": "test-model"}]}
    
    # Call the function that starts the server
    dolphin_server_module.run_server_main()
    
    mock_load_config.assert_called_once()
    mock_app_run.assert_called_once() # Check if Flask's app.run() is called
    assert len(dolphin_server_module.MODEL_CONFIG) == 1 # Check if MODEL_CONFIG was set

@patch('dolphin_logger.server.load_config', side_effect=Exception("Config load failed"))
@patch('dolphin_logger.server.app.run')
def test_run_server_main_config_load_exception(mock_app_run, mock_load_config_exc):
    # Call the function that starts the server
    dolphin_server_module.run_server_main() # Should still try to run server but with empty MODEL_CONFIG
    
    mock_load_config_exc.assert_called_once()
    assert dolphin_server_module.MODEL_CONFIG == [] # Ensure MODEL_CONFIG is empty on error
    mock_app_run.assert_called_once() # Server should still attempt to run

# Test the main proxy dispatching to core_proxy handlers
@patch('dolphin_logger.server.handle_anthropic_sdk_request')
@patch('dolphin_logger.server.get_target_api_config')
def test_proxy_dispatches_to_anthropic_handler(mock_get_config, mock_anthropic_handler, client, mock_model_config_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server
    
    mock_get_config.return_value = {
        "target_api_url": "anthropic_sdk",
        "target_api_key": "anthropic_key_from_config",
        "target_model": "claude-provider-model", # This is the providerModel
        "provider": "anthropic", "error": None
    }
    # Simulate a successful response from the handler
    mock_anthropic_handler.return_value = flask_app.response_class(
        response=json.dumps({"id": "anthropic_resp_id", "content": "response from mock anthropic"}),
        status=200, mimetype='application/json'
    )
        
    request_payload = {"model": "anthropic-claude3-server", "messages": [{"role": "user", "content": "Hi"}], "stream": False}
    response = client.post('/v1/chat/completions', json=request_payload)
    
    assert response.status_code == 200
    mock_get_config.assert_called_once_with("anthropic-claude3-server", mock_model_config_for_server)
    
    # Check arguments passed to the handler
    # json_data_for_sdk will have 'model' changed to 'claude-provider-model'
    expected_json_data_for_sdk = request_payload.copy()
    expected_json_data_for_sdk['model'] = "claude-provider-model"

    mock_anthropic_handler.assert_called_once_with(
        json_data_for_sdk=expected_json_data_for_sdk,
        target_model="claude-provider-model",
        target_api_key="anthropic_key_from_config",
        is_stream=False,
        original_request_json_data=request_payload
    )
    assert response.json["id"] == "anthropic_resp_id"


@patch('dolphin_logger.server.handle_rest_api_request')
@patch('dolphin_logger.server.get_target_api_config')
def test_proxy_dispatches_to_rest_handler(mock_get_config, mock_rest_handler, client, mock_model_config_for_server):
    dolphin_server_module.MODEL_CONFIG = mock_model_config_for_server
    
    mock_get_config.return_value = {
        "target_api_url": "http://localhost:12345/v1", # Some REST API base
        "target_api_key": "rest_api_key_from_config",
        "target_model": "gpt-4-provider-model", # providerModel
        "provider": "openai", "error": None
    }
    mock_rest_handler.return_value = flask_app.response_class(
        response=json.dumps({"id": "rest_resp_id", "content": "response from mock rest"}),
        status=200, mimetype='application/json'
    )
        
    request_payload = {"model": "openai-gpt4-server", "messages": [{"role": "user", "content": "Hi"}], "stream": False}
    response = client.post('/v1/chat/completions/some/path', json=request_payload) # Added /some/path
    
    assert response.status_code == 200
    mock_get_config.assert_called_once_with("openai-gpt4-server", mock_model_config_for_server)

    expected_json_data_for_downstream = request_payload.copy()
    expected_json_data_for_downstream['model'] = "gpt-4-provider-model"
    expected_data_bytes = json.dumps(expected_json_data_for_downstream).encode('utf-8')
    
    # Check specific args for handle_rest_api_request
    called_args, called_kwargs = mock_rest_handler.call_args
    assert called_kwargs['method'] == 'POST'
    assert called_kwargs['url'] == "http://localhost:12345/v1/chat/completions/some/path" # Path appended correctly
    assert called_kwargs['data_bytes'] == expected_data_bytes
    assert called_kwargs['is_stream'] == False
    assert called_kwargs['original_request_json_data'] == request_payload
    assert 'Authorization' in called_kwargs['headers']
    assert called_kwargs['headers']['Authorization'] == 'Bearer rest_api_key_from_config'
    
    assert response.json["id"] == "rest_resp_id"

```
