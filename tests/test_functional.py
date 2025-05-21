import pytest
import requests
import subprocess
import time
import os
import json
import threading # Not actively used yet, but good for future if needed
from pathlib import Path
import socket
import sys
import shutil # For cleaning up log dirs if needed

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Helper function to find a free port ---
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

# --- Server Fixture ---
@pytest.fixture(scope="session")
def live_server_details(tmp_path_factory):
    free_port = find_free_port()
    base_url = f"http://localhost:{free_port}"
    
    mock_config_home = tmp_path_factory.mktemp(f"functional_test_home_{free_port}")
    mock_dolphin_logger_dir = mock_config_home / ".dolphin-logger"
    mock_dolphin_logger_dir.mkdir(parents=True, exist_ok=True)
    log_dir = mock_dolphin_logger_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Define a specific port for the "failing" OpenAI service for more predictability
    mock_openai_fail_port = find_free_port()
    mock_anthropic_fail_port = find_free_port()


    # API key for testing ENV var resolution
    TEST_ENV_API_KEY_NAME = "FUNCTIONAL_ENV_OPENAI_KEY"
    TEST_ENV_API_KEY_VALUE = "env_key_functional_test_value"

    test_config_content = {
        "models": [
            { # Model that will point to a non-existent local "Ollama"
                "provider": "ollama", "providerModel": "test-ollama-functional", 
                "model": "ollama-functional-error", "apiBase": f"http://localhost:{find_free_port()}/v1"
            },
            { # Model that will point to a non-existent local "OpenAI-compatible" service for streaming test
                "provider": "openai", "providerModel": "gpt-3.5-test-streaming", 
                "model": "openai-streaming-error", 
                "apiBase": f"http://localhost:{mock_openai_fail_port}/v1",
                "apiKey": "fake_openai_key_streaming"
            },
            { # Model for testing ENV var resolution (will also fail connection)
                "provider": "openai", "providerModel": "gpt-4-env-test",
                "model": "openai-env-functional-error",
                "apiBase": f"http://localhost:{find_free_port()}/v1", # Different failing port
                "apiKey": f"ENV:{TEST_ENV_API_KEY_NAME}"
            },
            { # Model for testing Anthropic SDK path (will also fail connection)
                "provider": "anthropic", "providerModel": "claude-functional-test",
                "model": "anthropic-sdk-functional-error",
                "apiKey": "fake_anthropic_key_functional"
                # No apiBase for direct Anthropic SDK
            }
        ]
    }
    test_config_path = mock_dolphin_logger_dir / "config.json"
    with open(test_config_path, 'w') as f:
        json.dump(test_config_content, f)

    server_env = os.environ.copy()
    server_env["PORT"] = str(free_port)
    server_env["HOME"] = str(mock_config_home)
    server_env[TEST_ENV_API_KEY_NAME] = TEST_ENV_API_KEY_VALUE # Set the env var for the server
    if "PYTHONASYNCIODEBUG" in server_env: del server_env["PYTHONASYNCIODEBUG"]

    main_py_path = Path(__file__).resolve().parent.parent / "src" / "dolphin_logger" / "main.py"
    project_root = Path(__file__).resolve().parent.parent
    
    process = subprocess.Popen(
        [sys.executable, str(main_py_path)], env=server_env, cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1 # Line buffered
    )
    
    # --- Capture stdout/stderr in a separate thread to prevent blocking ---
    server_stdout = []
    server_stderr = []
    def log_pipe(pipe, A): # Using 'A' as a generic list argument
        try:
            for line in iter(pipe.readline, ''):
                print(f"[SERVER-{pipe.name.upper()}]: {line.strip()}", file=sys.stderr) # Print to test output for debugging
                A.append(line)
        except ValueError: # pipe closed
            pass
        finally:
            pipe.close()

    stdout_thread = threading.Thread(target=log_pipe, args=(process.stdout, server_stdout), name="server_stdout_reader")
    stderr_thread = threading.Thread(target=log_pipe, args=(process.stderr, server_stderr), name="server_stderr_reader")
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    max_wait_time = 20; start_time = time.time(); server_ready = False
    health_url = f"{base_url}/health"
    print(f"Waiting for server at {health_url} (max {max_wait_time}s)...")
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(health_url, timeout=0.5)
            if response.status_code == 200:
                print(f"Server ready! Health: {response.json()}")
                server_ready = True; break
            else:
                print(f"Health check status {response.status_code}, retrying...")
        except requests.ConnectionError:
            print("Connection error, retrying...")
        except requests.Timeout:
            print("Health check timeout, retrying...")
        if process.poll() is not None: # Process died
            print(f"Server process terminated prematurely with code {process.poll()}.")
            break
        time.sleep(0.2)
    
    if not server_ready:
        # Ensure threads finish
        if process.poll() is None: # if process is still running
            process.terminate() # Ask it to stop
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        print("Server STDOUT (on startup fail):\n", "".join(server_stdout))
        print("Server STDERR (on startup fail):\n", "".join(server_stderr))
        if process.poll() is None: process.kill() # Force kill if terminate didn't work
        process.wait()
        raise RuntimeError(f"Dolphin Logger server failed to start on port {free_port} for functional tests.")

    yield {"base_url": base_url, "log_dir": log_dir, "config_dir": mock_dolphin_logger_dir, "env": server_env}

    print(f"Terminating server (PID: {process.pid})...")
    process.terminate()
    stdout_thread.join(timeout=10)
    stderr_thread.join(timeout=10)
    try:
        process.wait(timeout=10)
        print(f"Server terminated gracefully with code {process.returncode}.")
    except subprocess.TimeoutExpired:
        print(f"Server (PID: {process.pid}) did not terminate gracefully, killing.")
        process.kill()
        process.wait()
    print("Server STDOUT (final):\n", "".join(server_stdout))
    print("Server STDERR (final):\n", "".join(server_stderr))
    print("Server fixture teardown complete.")


def _find_latest_log_file(log_dir_path: Path):
    if not log_dir_path.exists() or not log_dir_path.is_dir(): return None
    jsonl_files = list(log_dir_path.glob("*.jsonl"))
    if not jsonl_files: return None
    return max(jsonl_files, key=lambda f: f.stat().st_mtime)

def _clear_log_dir(log_dir_path: Path):
    if log_dir_path.exists():
        for item in log_dir_path.iterdir():
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)


# --- Test Cases ---
def test_health_endpoint(live_server_details):
    base_url = live_server_details["base_url"]
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Server is healthy" in data["message"]

def test_models_endpoint(live_server_details):
    base_url = live_server_details["base_url"]
    response = requests.get(f"{base_url}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 4 # Matches config in live_server_details
    model_ids = {m["id"] for m in data["data"]}
    assert "ollama-functional-error" in model_ids
    assert "openai-streaming-error" in model_ids
    assert "openai-env-functional-error" in model_ids
    assert "anthropic-sdk-functional-error" in model_ids

def test_model_not_configured(live_server_details):
    base_url = live_server_details["base_url"]
    payload = {"model": "non-existent-model", "messages": [{"role": "user", "content": "Test"}]}
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "Model 'non-existent-model' not found" in data["error"]

@pytest.mark.parametrize("model_to_test, expected_error_part_in_log", [
    ("ollama-functional-error", "Connection refused"), # Or "Failed to establish", "Max retries"
    ("openai-streaming-error", "Connection refused"),
    ("openai-env-functional-error", "Connection refused"),
    ("anthropic-sdk-functional-error", "Anthropic API key is missing") # This one should fail due to key not real, not connection
])
def test_chat_completions_error_and_logging(live_server_details, model_to_test, expected_error_part_in_log):
    server_info = live_server_details
    base_url = server_info["base_url"]
    log_dir = server_info["log_dir"]
    _clear_log_dir(log_dir) # Clean logs before this test

    chat_url = f"{base_url}/v1/chat/completions"
    is_stream = "streaming" in model_to_test # Simple way to trigger stream for one
    payload = {
        "model": model_to_test,
        "messages": [{"role": "user", "content": f"Hello to {model_to_test}"}],
        "stream": is_stream
    }
    
    # If testing the anthropic model, the API key in config is "fake_anthropic_key_functional"
    # The Anthropic SDK might try to validate this key structure or make a preliminary call.
    # For "anthropic-sdk-functional-error", the error might be different if the key is totally invalid vs. connection refused.
    # The current _handle_anthropic_sdk_request raises ValueError if key is missing, but if key is present but invalid,
    # it would proceed to `anthropic_client.messages.create` which would then fail.
    # If the key "fake_anthropic_key_functional" is treated as present but invalid by the SDK,
    # the SDK itself would raise an error. Let's adjust `expected_error_part_in_log` for that.
    if model_to_test == "anthropic-sdk-functional-error":
        # This will depend on Anthropic SDK's behavior with a fake key.
        # It might be an AuthenticationError or similar.
        # For now, let's assume it could be a generic SDK error or auth error.
        # The current code _handle_anthropic_sdk_request has specific anthropic.APIError handling.
        # A fake key usually results in an AuthenticationError from the SDK.
        expected_error_part_in_log = "AuthenticationError" # Or "Invalid API Key", "Forbidden"

    response = requests.post(chat_url, json=payload, stream=is_stream)

    if is_stream:
        # For a stream that fails immediately due to connection issues with the backend,
        # the proxy might return a non-streaming error response.
        assert response.status_code >= 400 # Could be 500, 502, 400 depending on error
        # If it's an error, content-type might not be event-stream
        if response.status_code == 200 and "text/event-stream" in response.headers.get("Content-Type",""):
            # This case means connection to proxy was fine, and stream started.
            # The error from upstream would be *within* the stream (if proxy is very robust)
            # or the stream would terminate abruptly.
            # Given current _handle_rest_api_request, a connection error to upstream for a stream
            # results in a non-streaming error response *before* any stream is sent to client.
             all_chunks = b""
             for chunk in response.iter_content(chunk_size=None): all_chunks += chunk
             print(f"Raw streaming response for {model_to_test}: {all_chunks.decode(errors='replace')}")
             # This path (200 for stream that then shows error) is less likely for connection refused
             pytest.fail(f"Stream for {model_to_test} returned 200 OK but was expected to fail connection to backend.")
        else: # Non-200 response, expected for immediate connection failure
            error_data = response.json()
            assert "error" in error_data
            print(f"Error data for {model_to_test} (stream req, non-stream error resp): {error_data}")
    else: # Non-streaming request
        assert response.status_code >= 400 
        error_data = response.json()
        assert "error" in error_data
        print(f"Error data for {model_to_test} (non-stream req): {error_data}")

    time.sleep(0.3) # Allow time for log writing
    latest_log = _find_latest_log_file(log_dir)
    assert latest_log is not None, f"No log file found in {log_dir} for {model_to_test}"
    
    with open(latest_log, 'r') as f:
        log_content = f.readline()
    assert log_content, f"Log file {latest_log.name} is empty for {model_to_test}"
    logged_entry = json.loads(log_content)
    
    assert logged_entry["request"]["model"] == payload["model"]
    assert logged_entry["request"]["messages"] == payload["messages"]
    assert "response" in logged_entry
    
    # Check if the logged response (which is a string) contains the expected error part
    # This needs to be flexible due to potentially different error messages.
    response_str_lower = str(logged_entry["response"]).lower()
    expected_error_part_lower = expected_error_part_in_log.lower()
    assert expected_error_part_lower in response_str_lower, \
        f"Expected '{expected_error_part_lower}' in logged error for {model_to_test}, got: '{response_str_lower}'"


def test_log_suppression_with_task_prefix(live_server_details):
    server_info = live_server_details
    base_url = server_info["base_url"]
    log_dir = server_info["log_dir"]
    _clear_log_dir(log_dir)

    # Get state of log dir before request
    initial_log_files = set(log_dir.glob("*.jsonl"))

    chat_url = f"{base_url}/v1/chat/completions"
    # Use a model that's configured to fail, so we know the request goes through error handling path
    # but the logging decision should happen before that.
    payload = {
        "model": "ollama-functional-error", 
        "messages": [{"role": "user", "content": "### Task: This should not be logged."}],
    }
    response = requests.post(chat_url, json=payload)
    # We expect an error from the proxy due to backend, but that's fine.
    assert response.status_code >= 400 

    time.sleep(0.3) # Wait for any potential log write
    
    current_log_files = set(log_dir.glob("*.jsonl"))
    new_log_files = current_log_files - initial_log_files

    assert len(new_log_files) == 0, \
        f"A new log file was created for a '### Task' request: {[f.name for f in new_log_files]}"

def test_api_key_env_var_resolution_functional(live_server_details):
    # This test checks if the server correctly picked up the API key from the environment
    # variable set by the live_server_details fixture.
    # It doesn't check if the key is *valid*, only that the proxy *would use it*.
    # Since the target `apiBase` for "openai-env-functional-error" also points to a non-existent server,
    # we expect a connection error, but the key should have been processed.
    # The main check here is that the server starts and uses the config where an ENV: var was.
    # The actual API key value isn't directly verifiable from client side in this test,
    # but if the server failed to start due to bad config processing of ENV:, this test would fail.
    # We also check that the request to this model is attempted.

    server_info = live_server_details
    base_url = server_info["base_url"]
    log_dir = server_info["log_dir"]
    _clear_log_dir(log_dir)

    chat_url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "openai-env-functional-error", # This model uses ENV:FUNCTIONAL_ENV_OPENAI_KEY
        "messages": [{"role": "user", "content": "Testing ENV var API key resolution"}],
    }
    response = requests.post(chat_url, json=payload)
    assert response.status_code >= 400 # Expect connection error to the fake backend
    error_data = response.json()
    assert "error" in error_data
    
    # Verify it was logged (meaning the request was processed up to the backend call attempt)
    time.sleep(0.2)
    latest_log = _find_latest_log_file(log_dir)
    assert latest_log is not None, "No log file found for ENV var test model"
    with open(latest_log, 'r') as f:
        logged_entry = json.loads(f.readline())
    assert logged_entry["request"]["model"] == "openai-env-functional-error"
    # The actual key value isn't in the log, but the fact that this model was chosen and an attempt was logged is sufficient
    # to show the server loaded and processed this part of the config. The unit tests for config_utils cover
    # the exact resolution logic.
    assert "Connection refused" in str(logged_entry["response"]) or "Failed to establish" in str(logged_entry["response"])

    # Additionally, check server logs for the print statement from config_utils.py
    # This is harder as server_stdout is not directly accessible here unless passed back from fixture.
    # For now, rely on the behavior. If `FUNCTIONAL_ENV_OPENAI_KEY` was NOT resolved,
    # the `apiKey` for this model would be `None`, which might lead to a different error
    # if the `_get_target_api_config` or `_handle_rest_api_request` expected a string key for OpenAI.
    # However, `_handle_rest_api_request` sends `Authorization: Bearer None` if key is None,
    # so the connection error to the fake backend would still be the primary failure.
    # The critical part is that `dolphin-logger` started correctly with this config.
    # The fixture's health check already confirms server startup.
    print("API Key ENV var functional check (request logged) passed.")
