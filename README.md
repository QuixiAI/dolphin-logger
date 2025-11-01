# dolphin-logger

![Dolphin Logger](chat-logger.png)

A proxy server that logs your LLM conversations to create datasets from your interactions with any Chat LLM. Supports multiple LLM backends including OpenAI, Anthropic, Google, and Ollama with full OpenAI API compatibility.

## Quick Start

1. **Install:** `pip install dolphin-logger` 
2. **Initialize:** `dolphin-logger init`
3. **Configure:** Edit your config file with your API keys (see path below)
4. **Run:** `dolphin-logger` (starts server on http://localhost:5001)
5. **Use:** Point your LLM client to `http://localhost:5001` instead of the original API

## Features

- Maintains OpenAI API compatibility
- Supports multiple LLM backends through configuration:
    - OpenAI (e.g., gpt-4.1)
    - Anthropic native SDK (e.g., claude-3-opus)
    - Anthropic via OpenAI-compatible API
    - Google (e.g., gemini-pro)
    - Ollama (local models e.g., codestral, dolphin)
    - Claude Code CLI (leverages Claude Max subscriptions)
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
- Command-line interface for server operation, log uploading, and configuration management.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cognitivecomputations/dolphin-logger.git
   cd dolphin-logger
   ```

2. **Install the package:**
   It's recommended to install in a virtual environment.
   
   **On Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install .
   ```
   
   **On Linux/Mac:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install .
   ```

## Configuration

Dolphin Logger uses a `config.json` file to define available LLM models and their settings.

**1. Initial Setup (First-Time Users):**

   Run the `init` command to create the necessary configuration directory and a default configuration file:
   ```bash
   dolphin-logger init
   ```
   This will:
   - Create the configuration directory if it doesn't exist.
   - Copy a `config.json.example` file from the package to your config directory.
   - If a `config.json` already exists, it will not be overwritten.

**2. Configuration File Location:**

   The active configuration file location depends on your operating system:
   - **Windows:** `%USERPROFILE%\.dolphin-logger\config.json`
   - **Linux/Mac:** `~/.dolphin-logger/config.json`

   You can check this path using the CLI:
   ```bash
   dolphin-logger config --path
   ```

**3. Editing `config.json`:**

   Open your configuration file and customize it with your desired models, API endpoints, and API keys. A `config.json.example` is included in the root of this repository for reference.

   *Example `config.json` structure:*
   ```json
   {
     "huggingface_repo": "cognitivecomputations/dolphin-logger",
     "models": [
       {
         "provider": "anthropic",
         "providerModel": "claude-3-7-sonnet-latest",
         "model": "claude",
         "apiKey": "ENV:ANTHROPIC_API_KEY"
       },
       {
         "provider": "openai",
         "providerModel": "gpt-4.1",
         "model": "gpt",
         "apiBase": "https://api.openai.com/v1/",
         "apiKey": "ENV:OPENAI_API_KEY"
       },
       {
         "provider": "openai",
         "providerModel": "gemini-2.5-pro-preview-05-06",
         "model": "gemini",
         "apiBase": "https://generativelanguage.googleapis.com/v1beta/",
         "apiKey": "ENV:GOOGLE_API_KEY"
       },
       {
         "provider": "ollama",
         "providerModel": "codestral:22b-v0.1-q5_K_M",
         "model": "codestral"
       },
       {
         "provider": "ollama",
         "providerModel": "dolphin3",
         "model": "dolphin"
       },
       {
         "provider": "claude_code",
         "providerModel": "unknown",
         "model": "claude-code"
       }
     ]
   }
   ```

Configuration fields:
- `provider`: The provider type:
  - "openai" for OpenAI-compatible APIs
  - "anthropic" for native Anthropic SDK (recommended for Claude models)
  - "ollama" for local Ollama models
  - "claude_code" for Claude Code CLI (leverages Claude Max subscriptions)
- `providerModel`: The actual model name to send to the provider's API
- `model`: The model name that clients will use when making requests to the proxy
- `apiBase`: The base URL for the API. For Ollama, this defaults to `http://localhost:11434/v1` if not specified. For Anthropic (using the native SDK via `provider: "anthropic"`), this field is not used. For Claude Code, this field is not used.
- `apiKey`: The API key for authentication. Not needed for Ollama or Claude Code. This can be the actual key string or a reference to an environment variable.

**Using Environment Variables for API Keys (Recommended for Security):**

To avoid hardcoding API keys in your `config.json`, you can instruct Dolphin Logger to read them from environment variables:
- In the `apiKey` field for a model, use the prefix `ENV:` followed by the name of the environment variable.
  For example: `"apiKey": "ENV:MY_OPENAI_API_KEY"`
- Dolphin Logger will then look for an environment variable named `MY_OPENAI_API_KEY` and use its value.
- If the specified environment variable is not set at runtime, a warning will be logged during startup, and the API key for that model will be treated as missing (effectively `None`). This might lead to authentication errors if the provider requires a key.

*Setting Environment Variables:*

**On Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=your_anthropic_key
set OPENAI_API_KEY=your_openai_key
set GOOGLE_API_KEY=your_google_key
```

**On Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your_anthropic_key"
$env:OPENAI_API_KEY="your_openai_key"
$env:GOOGLE_API_KEY="your_google_key"
```

**On Linux/Mac:**
```bash
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_google_key
```

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
- If `apiBase` is not specified for an Ollama provider, it defaults to `http://localhost:11434/v1`.
- No API key is required for local Ollama models.

Note for Claude Code:
- The "claude_code" provider uses the Claude Code CLI to leverage your Claude Max subscription
- Requires Claude Code to be installed and authenticated (run `claude setup-token`)
- No API key is required in the config - authentication is handled by Claude Code CLI
- The `providerModel` field is not used (Claude Code manages model selection internally)
- All requests are logged with detailed usage and cost information from Claude Code

**4. Validate Configuration (Optional):**
   After editing, you can validate your `config.json`:
   ```bash
   dolphin-logger config --validate
   ```
   This will check for JSON syntax errors and basic structural issues.

## Command-Line Interface (CLI)

Dolphin Logger is managed via the `dolphin-logger` command-line tool.

```
dolphin-logger [command] [options]
```

**Available Commands:**

*   **`server` (default)**
    *   Starts the LLM proxy server.
    *   This is the default action if no command is specified.
    *   Example:
        ```bash
        dolphin-logger
        # or explicitly
        dolphin-logger server
        # or with custom port
        dolphin-logger server --port 8080
        ```
    *   The server will run on port 5001 by default (configurable via the `PORT` environment variable or `--port` flag).

*   **`upload`**
    *   Uploads collected `.jsonl` log files from your logs directory to a configured Hugging Face Hub dataset repository.
    *   Example:
        ```bash
        dolphin-logger upload
        ```
    *   See the "Uploading Logs" section for prerequisites and details.

*   **`init`**
    *   Initializes the Dolphin Logger configuration.
    *   Creates the configuration directory if it doesn't exist.
    *   Copies a default `config.json.example` to your config directory if no `config.json` is present. This file serves as a template for your actual configuration.
    *   Example:
        ```bash
        dolphin-logger init
        ```

*   **`config`**
    *   Manages or inspects the configuration.
    *   `--path`: Shows the absolute path to the `config.json` file (shell-friendly output).
        ```bash
        dolphin-logger config --path
        ```
        *Powerful shell operations using the path:*
        ```bash
        # Edit config file directly
        vim $(dolphin-logger config --path)
        
        # Copy config to backup
        cp $(dolphin-logger config --path) backup-config.json
        
        # Check if config exists
        [ -f "$(dolphin-logger config --path)" ] && echo "Config exists"
        
        # View config contents
        cat $(dolphin-logger config --path)
        
        # Use in scripts
        CONFIG_PATH=$(dolphin-logger config --path)
        echo "Config is at: $CONFIG_PATH"
        ```
    *   `--validate`: Attempts to load and validate the `config.json` file, checking for JSON correctness and basic structure. It will also report on API keys resolved from environment variables.
        ```bash
        dolphin-logger config --validate
        ```

## Using the Proxy Server

Once the server is running (using `dolphin-logger` or `dolphin-logger server`):

1.  **List available models:**
    You can check the available models by calling the `/v1/models` endpoint:
    ```bash
    curl http://localhost:5001/v1/models 
    ```
    This will return a list of models as defined in your configuration file.

2.  **Make chat completion requests:**
    Use the proxy as you would the OpenAI API, but point your client's base URL to `http://localhost:5001` (or your configured port). Include the model name (as defined in the `model` field in your `config.json`) in your request.

    *cURL example using a model named "claude":*
    ```bash
    curl http://localhost:5001/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dummy-token" \
      -d '{
        "model": "claude",
        "messages": [{"role": "user", "content": "Hello from Claude!"}],
        "stream": false
      }'
    ```

    *Python OpenAI SDK example:*
    ```python
    from openai import OpenAI

    # Point to your local dolphin-logger proxy
    client = OpenAI(
        base_url="http://localhost:5001/v1",
        api_key="dummy-key"  # Not validated by proxy
    )

    response = client.chat.completions.create(
        model="claude",  # Use model name from your config
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
    ```

    *Using with popular tools:*
    - **Cursor/Continue.dev:** Set API base URL to `http://localhost:5001/v1`
    - **LangChain:** Use `openai_api_base="http://localhost:5001/v1"`
    - **Any OpenAI-compatible client:** Point base URL to your proxy

3.  **Check Server Health:**
    Verify server status and configuration load:
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
    *If configuration issues exist (e.g., no models loaded):*
    ```json
    {
      "status": "error",
      "message": "Server is running, but configuration might have issues (e.g., no models loaded)."
    }
    ```

## Environment Variables

The proxy primarily uses the following environment variables:

- `PORT`: Sets the port on which the proxy server will listen (default: `5001`).
- **API Keys (Dynamic)**: Any environment variable that you reference in your `config.json` using the `ENV:` prefix for `apiKey` fields (e.g., if you have `"apiKey": "ENV:OPENAI_API_KEY"`, then `OPENAI_API_KEY` becomes a relevant environment variable).
- `HF_TOKEN` (for uploading logs): A Hugging Face Hub token with write access to the target dataset repository is required when using the `dolphin-logger upload` command.

## Logging

- All proxied requests and their corresponding responses to `/v1/chat/completions` are automatically logged.
- Logs are stored in date-specific `.jsonl` files (one line per JSON object).
- Log files are named with UUIDs to ensure uniqueness (e.g., `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.jsonl`).
- Logging is thread-safe.
- **Log Suppression:** If the first user message in a request starts with `### Task:`, that specific request/response pair will not be logged. This is useful for excluding specific interactions (e.g., meta-prompts or system commands) from your dataset.

**Log File Locations:**
- **Windows:** `%USERPROFILE%\.dolphin-logger\logs\`
- **Linux/Mac:** `~/.dolphin-logger/logs/`

## Uploading Logs

The `dolphin-logger upload` command facilitates uploading your collected logs to a Hugging Face Hub dataset.

**Prerequisites:**
- **Hugging Face Token:** Ensure the `HF_TOKEN` environment variable is set with a Hugging Face token. This token must have write access to the target dataset repository.
- **Target Repository:** The default target dataset is `cognitivecomputations/dolphin-logger`. This can be changed by modifying the `huggingface_repo` field in your `config.json`.

**Process:**
1.  Run: `dolphin-logger upload`
2.  The tool will:
    *   Locate all `.jsonl` files in your logs directory.
    *   Create a new branch in the target Hugging Face dataset (e.g., `upload-logs-YYYYMMDD-HHMMSS`).
    *   Commit the log files to this new branch.
    *   Attempt to create a Pull Request (Draft) from this new branch to the dataset's main branch.
    *   Print the URL of the created Pull Request.
3.  You will need to visit the URL to review and merge the Pull Request on Hugging Face Hub.

## Troubleshooting

**Common Issues and Solutions:**

1. **"Configuration file not found" error:**
   - Run `dolphin-logger init` to create the default configuration
   - Check that your config file exists with `dolphin-logger config --path`

2. **"Template config file not found" error:**
   - Ensure you've installed the package properly with `pip install .`
   - Verify the `config.json.example` file exists in the project root

3. **API authentication errors:**
   - Verify your API keys are correctly set in environment variables
   - Check that environment variable names match those specified in config (e.g., `ENV:OPENAI_API_KEY` requires `OPENAI_API_KEY` to be set)
   - Use `dolphin-logger config --validate` to check API key resolution

4. **Server won't start / Port already in use:**
   - **Windows:** Check if another process is using port 5001: `netstat -an | findstr :5001`
   - **Linux/Mac:** Check if another process is using port 5001: `lsof -i :5001`
   - Set a different port: `dolphin-logger server --port 5002` or `set PORT=5002` (Windows) / `export PORT=5002` (Linux/Mac)
   - Kill existing processes if needed

5. **Models not appearing in `/v1/models` endpoint:**
   - Validate your configuration: `dolphin-logger config --validate`
   - Check that your config.json has a properly formatted "models" array
   - Restart the server after configuration changes

6. **Ollama models not working:**
   - Ensure Ollama is running: `ollama list`
   - Check that the model names in your config match available Ollama models
   - Verify Ollama is accessible at `http://localhost:11434`

7. **Claude Code models not working:**
   - Ensure Claude Code is installed: `claude --version`
   - Verify authentication: `claude setup-token`
   - Test Claude Code directly: `echo "test" | claude chat --print`
   - Check server logs for specific error messages from Claude Code CLI

8. **Logs not being created:**
   - Check that requests don't start with "### Task:" (these are suppressed by default)
   - Verify the logs directory exists and is writable
   - Look for error messages in the server output

9. **HTTPS configuration error:**
   - If you see an HTTPS error message, change your client configuration from `https://localhost` to `http://localhost`
   - Dolphin Logger runs on HTTP, not HTTPS

**Getting Help:**
- Enable verbose logging with detailed error messages
- Check the server console output for specific error details
- Validate your configuration with `dolphin-logger config --validate`
- Ensure all required environment variables are set

## Error Handling

The proxy includes comprehensive error handling:
- Preserves original error messages from upstream APIs when available.
- Provides detailed error information in JSON format for debugging.
- Maintains appropriate HTTP status codes for different error types.
- Smart detection of HTTPS misconfiguration with helpful guidance.

## Project Structure

The `dolphin-logger` codebase is organized into several modules within the `src/dolphin_logger/` directory:

- `cli.py`: Handles command-line argument parsing and dispatches to appropriate functions for different commands (`server`, `upload`, `init`, `config`).
- `server.py`: Contains the Flask application setup, route definitions (`/health`, `/v1/models`, and the main proxy route), and the main server running logic.
- `core_proxy.py`: Implements the core logic for proxying requests. This includes selecting the target API based on configuration, and separate handlers for Anthropic SDK requests and general REST API requests.
- `providers/`: Provider-specific implementations for different LLM backends.
  - `claude_code.py`: Claude Code CLI provider implementation for leveraging Claude Max subscriptions.
- `config.py`: Manages configuration loading, creation of default configuration, and resolution of API keys from environment variables.
- `logging_utils.py`: Provides utilities for managing log files (daily rotation, UUID naming) and determining if a request should be logged.
- `upload.py`: Contains the logic for finding log files and uploading them to a Hugging Face Hub dataset.
- `main.py`: The primary entry point for the `dolphin-logger` script, which calls the main CLI function from `cli.py`.
- `__init__.py`: Makes `dolphin_logger` a Python package.

## Testing

The project includes a suite of unit and functional tests to ensure reliability and prevent regressions.

**Tools Used:**
- `pytest`: For test discovery and execution.
- `pytest-mock`: For mocking dependencies in unit tests.
- Standard Python `unittest.mock` and `subprocess` modules.

**Running Tests:**
1.  Ensure you have installed the development dependencies:
    ```bash
    pip install pytest pytest-mock requests
    ```
2.  Navigate to the root directory of the project.
3.  Run the tests:
    ```bash
    python -m pytest tests/
    ```
    Or more simply:
    ```bash
    pytest tests/
    ```

**Test Environment Notes:**
- **Unit Tests (`tests/test_*.py` excluding `test_functional.py`):** These are self-contained and mock external dependencies like file system operations, API calls, and environment variables. They test individual modules in isolation.
- **Functional Tests (`tests/test_functional.py`):**
    - These tests start an actual instance of the Dolphin Logger server on a free port using a subprocess.
    - They use a temporary, isolated configuration directory and log directory for each test session, ensuring they **do not interfere with your user-level configuration setup.**
    - The functional tests primarily verify the server's behavior with configurations that point to **non-existent backend services.** This allows testing of the proxy's routing, error handling, and logging mechanisms when upstream services are unavailable, without requiring actual LLM API keys or running local LLMs during the test execution.

## License

MIT
