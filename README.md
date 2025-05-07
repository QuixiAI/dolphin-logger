# dolphin-logger

![Dolphin Logger](chat-logger.png)

This proxy allows you to record your chats to create datasets based on your conversations with any Chat LLM. It now supports multiple LLM backends including OpenAI, Anthropic, and Google.

## Features

- Maintains OpenAI API compatibility
- Supports multiple LLM backends:
    - OpenAI (e.g., gpt-4, gpt-3.5-turbo)
    - Anthropic (e.g., claude-2, claude-instant-1)
    - Google (e.g., gemini-pro)
- Dynamic API key and endpoint selection based on the requested model in the `/v1/chat/completions` endpoint.
- Provides a `/v1/models` endpoint listing available models: "gpt", "claude", "gemini".
- Supports both streaming and non-streaming responses.
- Automatic request logging to JSONL format.
- Error handling with detailed responses.
- Request/response logging with thread-safe implementation.

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.local`. You'll need to provide API keys and endpoint URLs for the models you intend to use:
```env
# General
PORT=5001

# For OpenAI models (e.g., "gpt")
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_MODEL=gpt-4 # Default model if "gpt" is chosen and no specific model is in the request

# For Anthropic models (e.g., "claude")
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_ENDPOINT=your_anthropic_openai_compatible_endpoint_here
ANTHROPIC_MODEL=claude-2 # Default model if "claude" is chosen

# For Google models (e.g., "gemini")
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_ENDPOINT=your_google_openai_compatible_endpoint_here
GOOGLE_MODEL=gemini-pro # Default model if "gemini" is chosen

# For uploading to Hugging Face Hub
HF_TOKEN=your_hugging_face_write_token
```
Note:
- Ensure your `HF_TOKEN` has write permissions to the target repository for uploading logs.
- `*_ENDPOINT` variables should point to OpenAI API-compatible endpoints.

## Usage

1. Start the server:
```bash
python proxy.py
```

2. The server will run on port 5001 by default (configurable via `PORT` environment variable).

3. **List available models:**
   You can check the available model backends by calling the `/v1/models` endpoint:
   ```bash
   curl http://localhost:5001/v1/models
   ```
   This will return a list like:
   ```json
   {
     "object": "list",
     "data": [
       { "id": "gpt", "object": "model", "created": 1677610602, "owned_by": "openai" },
       { "id": "claude", "object": "model", "created": 1677610602, "owned_by": "anthropic" },
       { "id": "gemini", "object": "model", "created": 1677610602, "owned_by": "google" }
     ]
   }
   ```

4. **Make chat completion requests:**
   Use the proxy as you would the OpenAI API, but point your requests to your local server. To use a specific backend (gpt, claude, or gemini), include its name as the model in your request. If the model in the request is a specific model ID (e.g., `gpt-4-turbo`), the proxy will try to infer the backend. If it cannot, or if no model is specified, it will use the `OPENAI_MODEL` by default.

   Example using the "claude" backend (which will use `ANTHROPIC_MODEL` if not specified further):
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \ # Bearer token is not strictly validated by this proxy but included for compatibility
     -d '{
       "model": "claude", # or "gpt", "gemini", or a specific model ID like "claude-3-opus-20240229"
       "messages": [{"role": "user", "content": "Hello from Claude!"}],
       "stream": true
     }'
   ```

   Example using a specific "gpt" model:
   ```bash
   curl http://localhost:5001/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your-api-key" \
     -d '{
       "model": "gpt-4-turbo", 
       "messages": [{"role": "user", "content": "Hello from GPT-4 Turbo!"}],
       "stream": false
     }'
   ```

## Environment Variables

The proxy uses the following environment variables:

- `PORT`: Server port (default: 5001).

**Model Specific Environment Variables:**

*   **OpenAI (for "gpt" models):**
    *   `OPENAI_API_KEY`: Your OpenAI API key.
    *   `OPENAI_ENDPOINT`: The base URL for the OpenAI API (e.g., `https://api.openai.com/v1`).
    *   `OPENAI_MODEL`: The default OpenAI model to use if "gpt" is selected or no specific model is provided in the request (e.g., `gpt-4`).

*   **Anthropic (for "claude" models):**
    *   `ANTHROPIC_API_KEY`: Your Anthropic API key.
    *   `ANTHROPIC_ENDPOINT`: The base URL for your Anthropic API (must be OpenAI compatible).
    *   `ANTHROPIC_MODEL`: The default Anthropic model to use if "claude" is selected (e.g., `claude-2`).

*   **Google (for "gemini" models):**
    *   `GOOGLE_API_KEY`: Your Google API key.
    *   `GOOGLE_ENDPOINT`: The base URL for your Google API (must be OpenAI compatible).
    *   `GOOGLE_MODEL`: The default Google model to use if "gemini" is selected (e.g., `gemini-pro`).

**Other Environment Variables:**

- `HF_TOKEN`: Your Hugging Face Hub token with write access, used by `upload.py` for uploading logs.

## Logging

All requests and responses for `/v1/chat/completions` are automatically logged to `logs.jsonl` in JSONL format. The logging is thread-safe and includes both request and response content.

## Uploading Logs

The `upload.py` script facilitates uploading your collected `.jsonl` log files to a Hugging Face Hub dataset.

**Prerequisites:**
- Ensure you have set the `HF_TOKEN` environment variable with a Hugging Face token that has write access to the target dataset repository.
- The target dataset repository ID is configured within the `upload.py` script (default: `cognitivecomputations/coding-collect`).

**How to use:**
1. Run the script from the root directory of the project:
```bash
python upload.py
```
2. The script will:
    - Find all `.jsonl` files in the current directory.
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
