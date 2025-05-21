# dolphin-logger

![Dolphin Logger](chat-logger.png)

This proxy allows you to record your chats to create datasets based on your conversations with any Chat LLM. It supports multiple LLM backends including OpenAI, Anthropic, Google and Ollama.

## Features

- Maintains OpenAI API compatibility
- Supports multiple LLM backends through configuration:
    - OpenAI (e.g., gpt-4.1)
    - Anthropic native SDK (e.g., claude-3-opus)
    - Anthropic via OpenAI-compatible API
    - Google (e.g., gemini-pro)
    - Ollama (local models e.g., codestral, dolphin)
- Configuration-based model definition using `config.json`
- Dynamic API endpoint selection based on the requested model in the `/v1/chat/completions` endpoint.
- Provides a `/v1/models` endpoint listing all configured models.
- Provides a `/health` endpoint for server status and configuration load checks.
- Supports both streaming and non-streaming responses.
- Automatic request logging to JSONL format.
- Logging suppression for specific tasks (requests starting with "### Task:").
- Error handling with detailed responses.
- Request/response logging with thread-safe implementation.
- Support for API keys via environment variables for enhanced security.

## Setup

1. Clone the repository
2. Install the package:
```bash
pip install .
```

3.  The `config.json` file should be placed in `~/.dolphin-logger/config.json`. Create the directory if it doesn't exist.

4. Create a `config.json` file with your model configurations:
```json
{
  "models": [
    {
      "provider": "openai",
      "providerModel": "gpt-4.1",
      "model": "gpt4.1",
      "apiBase": "https://api.openai.com/v1",
      "apiKey": "your_openai_api_key_here"
    },
    {
      "provider": "anthropic",
      "providerModel": "claude-3-7-sonnet-latest",
      "model": "claude",
      "apiKey": "your_anthropic_api_key_here"
    },
    {
      "provider": "openai",
      "providerModel": "claude-3-7-sonnet-latest",
      "model": "claude-hyprlab",
      "apiBase": "https://api.hyprlab.io/private",
      "apiKey": "your_anthropic_api_key_here"
    },
    {
      "provider": "openai",
      "providerModel": "gemini-2.5-pro-preview-05-06",
      "model": "gemini",
      "apiBase": "https://generativelanguage.googleapis.com/v1beta/",
      "apiKey": "your_google_api_key_here"
    },
    {
      "provider": "ollama",
      "providerModel": "codestral:22b-v0.1-q5_K_M",
      "model": "codestral"
    },
    {
      "provider": "ollama",
      "providerModel": "dolphin3-24b",
      "model": "dolphin"
    }
  ]
}
```

Configuration fields:
- `provider`: The provider type:
  - "openai" for OpenAI-compatible APIs
  - "anthropic" for native Anthropic SDK (recommended for Claude models)
  - "ollama" for local Ollama models
- `providerModel`: The actual model name to send to the provider's API
- `model`: The model name that clients will use when making requests to the proxy
- `apiBase`: The base URL for the API. For Ollama, this defaults to `http://localhost:11434/v1` if not specified. For Anthropic (using the native SDK via `provider: "anthropic"`), this field is not used.
- `apiKey`: The API key for authentication. Not needed for Ollama. This can be the actual key string or a reference to an environment variable.

**Using Environment Variables for API Keys (Recommended for Security):**

To avoid hardcoding API keys in your `config.json`, you can instruct Dolphin Logger to read them from environment variables:
- In the `apiKey` field for a model, use the prefix `ENV:` followed by the name of the environment variable.
  For example: `"apiKey": "ENV:MY_OPENAI_API_KEY"`
- Dolphin Logger will then look for an environment variable named `MY_OPENAI_API_KEY` and use its value.
- If the specified environment variable is not set at runtime, a warning will be logged during startup, and the API key for that model will be treated as missing (effectively `None`). This might lead to authentication errors if the provider requires a key.

*Benefits:*
  - **Enhanced Security:** Keeps sensitive API keys out of configuration files, which might be accidentally committed to version control.
  - **Flexibility:** Allows different API keys for different environments (development, staging, production) without changing `config.json`. Ideal for Docker deployments and CI/CD pipelines.

*Example `config.json` entry:*
```json
    {
      "provider": "openai",
      "providerModel": "gpt-4-turbo",
      "model": "gpt-4-turbo-secure",
      "apiBase": "https://api.openai.com/v1",
      "apiKey": "ENV:OPENAI_API_KEY" 
    }
```
In this case, you would set the `OPENAI_API_KEY` environment variable in your system before running Dolphin Logger.

Note for Anthropic models:
- Using the "anthropic" provider is recommended as it uses the official Anthropic Python SDK
- This provides better performance and reliability compared to using Claude through an OpenAI-compatible API

Note for Ollama models:
- The proxy automatically uses "http://localhost:11434/v1" as the endpoint
- No API key is required for local Ollama models

## Usage

1. Start the server:
```bash
dolphin-logger
```

2. The server will run on port 5001 by default (configurable via `PORT` environment variable).

3. **List available models:**
   You can check the available models by calling the `/v1/models` endpoint:
   ```bash
   curl http://localhost:5001/v1/models
   ```
   This will return a list of models defined in your config.json:
   ```json
   {
     "object": "list",
     "data": [
       { "id": "gpt4.1", "object": "model", "created": 1686935002, "owned_by": "openai", "provider": "openai", "provider_model": "gpt-4.1" },
       { "id": "claude", "object": "model", "created": 1686935002, "owned_by": "openai", "provider": "openai", "provider_model": "claude-3-7-sonnet-latest" },
       { "id": "gemini", "object": "model", "created": 1686935002, "owned_by": "openai", "provider": "openai", "provider_model": "gemini-2.5-pro-preview-05-06" },
       { "id": "codestral", "object": "model", "created": 1686935002, "owned_by": "ollama", "provider": "ollama", "provider_model": "codestral:22b-v0.1-q5_K_M" },
       { "id": "dolphin", "object": "model", "created": 1686935002, "owned_by": "ollama", "provider": "ollama", "provider_model": "dolphin3-24b" }
     ]
   }
   ```

4. **Make chat completion requests:**
   Use the proxy as you would the OpenAI API, but point your requests to your local server. Include the model name (as defined in the `model` field in config.json) in your request.

   Example using the "claude" model (assuming "claude" is a defined model in your `config.json`):
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer any-valid-or-dummy-token" \ # Bearer token is not validated by the proxy but included for client compatibility
     -d '{
       "model": "claude",
       "messages": [{"role": "user", "content": "Hello from Claude!"}],
       "stream": true
     }'
   ```

   Example using a local Ollama model (assuming "dolphin" is a defined model in your `config.json` pointing to an Ollama provider):
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "dolphin", 
       "messages": [{"role": "user", "content": "Hello from Dolphin!"}],
       "stream": false
     }'
   ```

5. **Check Server Health:**
   You can verify if the server is running and if its configuration was loaded correctly using the `/health` endpoint:
   ```bash
   curl http://localhost:5001/health
   ```
   *Expected response (healthy):*
   ```json
   {
     "status": "ok",
     "message": "Server is healthy, configuration loaded."
   }
   ```
   *Expected response (if config has issues, e.g., no models loaded):*
   ```json
   {
     "status": "error",
     "message": "Server is running, but configuration might have issues (e.g., no models loaded)."
   }
   ```

## Environment Variables

The proxy uses the following environment variables:

- `PORT`: Server port (default: 5001).
- **API Keys**: Any environment variable referenced in `config.json` using the `ENV:` prefix for `apiKey` fields (e.g., `OPENAI_API_KEY` if you have `"apiKey": "ENV:OPENAI_API_KEY"` in your config).

## Logging

All requests and responses for `/v1/chat/completions` are automatically logged to date-specific `.jsonl` files with UUID-based names. The logging is thread-safe and includes both request and response content.

## Uploading Logs

You can upload your collected `.jsonl` log files to a Hugging Face Hub dataset using the built-in CLI:

**Prerequisites:**
- Ensure you have set the `HF_TOKEN` environment variable with a Hugging Face token that has write access to the target dataset repository.
- The target dataset repository ID is configured in the code (default: `cognitivecomputations/dolphin-logger`).

**How to use:**
1. Run the following command from anywhere:
```bash
dolphin-logger --upload
```
2. The CLI will:
    - Find all `.jsonl` files in your logs directory (`~/.dolphin-logger/logs/`).
    - Create a new branch in the specified Hugging Face dataset repository.
    - Commit the log files to this new branch.
    - Create a Pull Request (Draft) from the new branch to the main branch of the dataset repository.
    - Print the URL of the created Pull Request.
3. You will then need to review and merge the Pull Request on Hugging Face Hub.

## Error Handling

The proxy includes comprehensive error handling that:
- Preserves original error messages from upstream APIs when available.
- Provides detailed error information for debugging.
- Maintains proper HTTP status codes.

## Testing

The project includes a suite of unit and functional tests to ensure reliability and prevent regressions.

**Tools Used:**
- `pytest`: For test discovery and execution.
- `pytest-mock`: For mocking dependencies in unit tests.
- Standard Python `unittest.mock` and `subprocess` modules.

**Running Tests:**
1. Ensure you have installed the development dependencies (if any, e.g., `pip install .[dev]` if a `dev` extra is defined in `pyproject.toml` containing `pytest` and `pytest-mock`). If not, install them manually:
   ```bash
   pip install pytest pytest-mock requests
   ```
2. Navigate to the root directory of the project.
3. Run the tests using `pytest`:
   ```bash
   python -m pytest tests/
   ```
   Or simply:
   ```bash
   pytest tests/
   ```

**Test Environment Notes:**
- **Unit Tests:** These are self-contained and do not require external services. They mock file system operations, environment variables, and external API calls.
- **Functional Tests:**
    - These tests start an actual instance of the Dolphin Logger server on a free port.
    - They use a temporary, isolated configuration directory and log directory for each test session, so they **do not interfere with your user-level `~/.dolphin-logger` setup.**
    - The functional tests are designed to verify the server's behavior with configurations that point to **non-existent backend services.** This allows testing of the proxy's routing, error handling, and logging mechanisms when upstream services are unavailable, without requiring actual LLM API keys or running local LLMs.

## License

MIT
