import os
import json
import requests
from dotenv import load_dotenv
from flask import Flask, request, Response, stream_with_context
from threading import Lock

log_lock = Lock()
app = Flask(__name__)
LOG_FILE = 'logs.jsonl'

load_dotenv()
OPENAI_API_URL = os.getenv('OPENAI_ENDPOINT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    
    if path.endswith('/models'):
        return Response(
            json.dumps([{
                "id": OPENAI_MODEL,
                "object": "model",
                "created": 1686935002,
                "owned_by": "organization-owner"
            }]),
            content_type='application/json'
        )
    
    # print("\n=== Incoming Request ===")
    # print(f"Method: {request.method}")
    # print(f"Path: {path}")
    # print(f"Headers: {dict(request.headers)}")
    # print(f"Raw Data: {request.get_data().decode('utf-8')}")
    base_path = '/' + OPENAI_API_URL.rstrip('/').split('/')[-1]
    
    # Strip out the first path component and replace with the base_path
    path_parts = path.split('/', 1)
    actual_path = path_parts[1] if len(path_parts) > 1 else ''
    
    # Remove the base_path from OPENAI_API_URL if it exists
    base_url = OPENAI_API_URL.rstrip('/').rsplit(base_path, 1)[0]
    url = f"{base_url}{base_path}/{actual_path}"
    print(f"Proxying request to: {url}")
    
    headers = {k: v for k, v in request.headers.items() if k != 'Host'}
    headers['Host'] = url.split('//')[-1].split('/')[0]
    headers['Authorization'] = f'Bearer {OPENAI_API_KEY}'

    data = request.get_data()
    json_data = json.loads(data.decode('utf-8')) if data else None
    
    if request.method == 'POST':
        json_data['model'] = OPENAI_MODEL
        data = json.dumps(json_data).encode('utf-8')
    
    is_stream = json_data.get('stream', False) if json_data else False
    
    # print("\n=== Outgoing Request ===")
    # print(f"URL: {url}")
    # print(f"Headers: {headers}")
    # print(f"Data: {data.decode('utf-8') if data else None}")

    try:
        response = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=data,
            stream=is_stream,
        )
        
        response.raise_for_status()

        if request.method == 'POST':
            if is_stream:
                def generate():
                    response_content = ''
                    for line in response.iter_lines():
                        if line:
                            if line.startswith(b'data: '):
                                yield line + b'\n\n'  # Added extra newline for SSE
                                line_data = line.decode('utf-8')[6:]
                                if line_data != '[DONE]':
                                    response_content += json.loads(line_data)['choices'][0]['delta'].get('content', '')

                    with log_lock:
                        with open(LOG_FILE, 'a') as log_file:
                            log_file.write(json.dumps({
                                'request': json_data,
                                'response': response_content
                            }) + '\n')

                return Response(stream_with_context(generate()), content_type=response.headers['Content-Type'])
            else:
                response_data = response.json()
                complete_response = response_data['choices'][0]['message']['content']

                with log_lock:
                    with open(LOG_FILE, 'a') as log_file:
                        log_file.write(json.dumps({
                            'request': json_data,
                            'response': complete_response
                        }) + '\n')

                return response_data
        else:
            if is_stream:
                return Response(stream_with_context(response.iter_content(chunk_size=None)), content_type=response.headers['Content-Type'])
            else:
                return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error proxying request: {e}")
        # Get the actual error details from the response if available
        if hasattr(e.response, 'json'):
            try:
                error_data = e.response.json()
                return Response(json.dumps(error_data), 
                              status=e.response.status_code, 
                              content_type='application/json')
            except json.JSONDecodeError:
                pass
        
        # Fallback error response
        error_message = str(e)
        status_code = e.response.status_code if hasattr(e, 'response') else 500
        return Response(json.dumps({
            "error": {
                "message": error_message,
                "type": type(e).__name__,
                "status": status_code
            }
        }), status=status_code, content_type='application/json')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(port=port)