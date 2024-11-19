# chat-logger

![Chat Logger](chat-logger.png)

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
```

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

## Logging

All requests and responses are automatically logged to `logs.jsonl` in JSONL format. The logging is thread-safe and includes both request and response content.

## Error Handling

The proxy includes comprehensive error handling that:
- Preserves original OpenAI error messages when available
- Provides detailed error information for debugging
- Maintains proper HTTP status codes

## License

MIT
