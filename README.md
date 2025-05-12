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
- Dynamic API endpoint selection based on the requested model in the `/v1/chat/completions` endpoint
- Provides a `/v1/models` endpoint listing all configured models
- Supports both streaming and non-streaming responses
- Automatic request logging to JSONL format
- Error handling with detailed responses
- Request/response logging with thread-safe implementation

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
- `apiBase`: The base URL for the API (not needed for "anthropic" or "ollama" providers)
- `apiKey`: The API key for authentication (not needed for "ollama" provider)

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

   Example using the "claude" model:
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer any-token" \ # Bearer token is not validated but included for compatibility
     -d '{
       "model": "claude",
       "messages": [{"role": "user", "content": "Hello from Claude!"}],
       "stream": true
     }'
   ```

   Example using a local Ollama model:
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "dolphin", 
       "messages": [{"role": "user", "content": "Hello from Dolphin!"}],
       "stream": false
     }'
   ```

## Environment Variables

The proxy now uses minimal environment variables:

- `PORT`: Server port (default: 5001)

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

## License

MIT
