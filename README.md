# dolphin-logger

![Dolphin Logger](chat-logger.png)

This proxy allows you to record your chats to create datasets based on your conversations with any Chat LLM.

## Features

- Maintains OpenAI API compatibility
- Supports both streaming and non-streaming responses
- Automatic request logging to JSONL format
- Configurable model selection
- Error handling with detailed responses
- Request/response logging with thread-safe implementation

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.local`:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# For uploading to Hugging Face Hub
HF_TOKEN=your_hugging_face_write_token
```
Note: Ensure your `HF_TOKEN` has write permissions to the target repository.

## Usage

1. Start the server:
```bash
python proxy.py
```

2. The server will run on port 5001 by default (configurable via PORT environment variable)

3. Use the proxy exactly as you would use the OpenAI API, but point your requests to your local server:
```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_ENDPOINT`: The base URL for the OpenAI API
- `OPENAI_MODEL`: The default model to use for requests
- `PORT`: Server port (default: 5001)
- `HF_TOKEN`: Your Hugging Face Hub token with write access, used by `upload.py`.

## Logging

All requests and responses are automatically logged to `logs.jsonl` in JSONL format. The logging is thread-safe and includes both request and response content.

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
- Preserves original OpenAI error messages when available
- Provides detailed error information for debugging
- Maintains proper HTTP status codes

## License

MIT
