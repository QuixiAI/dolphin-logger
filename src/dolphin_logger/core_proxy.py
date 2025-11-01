import json
from flask import Response

from .providers.anthropic import handle_anthropic_request
from .providers.openai import handle_openai_request
from .providers.google import handle_google_request
from .providers.ollama import handle_ollama_request
from .providers.claude_code import handle_claude_code_request
# MODEL_CONFIG is typically loaded in server.py and passed to get_target_api_config.

def get_target_api_config(requested_model_id: str, model_config_list: list) -> dict:
    """
    Determines the target API URL, key, and model based on the requested model ID
    and the server's model configuration.

    Args:
        requested_model_id (str): The model ID from the client's request.
        model_config_list (list): The MODEL_CONFIG list (passed from server.py).

    Returns:
        dict: A dictionary containing 'target_api_url', 'target_api_key',
              'target_model', 'provider', and 'error' (if any).
    """
    for model_config in model_config_list:
        if model_config.get("model") == requested_model_id:
            provider = model_config.get("provider")
            target_model = model_config.get("providerModel", requested_model_id)
            api_key = model_config.get("apiKey")
            api_base = model_config.get("apiBase")

            if provider == "ollama":
                return {
                    "target_api_url": api_base or "http://localhost:11434/v1",
                    "target_api_key": "", 
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }
            elif provider == "anthropic":
                return {
                    "target_api_url": "anthropic_sdk", # Special marker
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }
            elif provider == "google":
                return {
                    "target_api_url": "google_sdk", # Special marker
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }
            elif provider == "claude_code":
                return {
                    "target_api_url": "claude_code_sdk", # Special marker
                    "target_api_key": None,  # Claude Code handles auth internally
                    "target_model": "claude-code",  # Placeholder - Claude Code handles model selection
                    "provider": provider,
                    "claude_code_path": model_config.get("claudeCodePath"),
                    "claude_code_oauth_token": model_config.get("claudeCodeOAuthToken"),
                    "max_output_tokens": model_config.get("maxOutputTokens"),
                    "error": None,
                }
            else: # OpenAI-compatible
                if not api_base:
                    # Ensure a specific error message for missing apiBase for OpenAI-like providers
                    return {"error": f"apiBase not configured for OpenAI-compatible model '{requested_model_id}'"}
                return {
                    "target_api_url": api_base,
                    "target_api_key": api_key,
                    "target_model": target_model,
                    "provider": provider,
                    "error": None,
                }

    if not model_config_list: # Check if the list itself is empty
        return {"error": f"No models configured. Cannot process request for model '{requested_model_id}'."}
    return {"error": f"Model '{requested_model_id}' not found in configured models."}


def handle_anthropic_sdk_request(
    json_data_for_sdk: dict, 
    target_model: str, 
    target_api_key: str | None, 
    is_stream: bool, 
    original_request_json_data: dict
) -> Response:
    """
    Handles requests to the Anthropic SDK.
    Delegates to the Anthropic provider.
    """
    return handle_anthropic_request(
        json_data_for_sdk, 
        target_model, 
        target_api_key, 
        is_stream, 
        original_request_json_data
    )


def handle_google_sdk_request(
    json_data_for_sdk: dict,
    target_model: str,
    target_api_key: str | None,
    is_stream: bool,
    original_request_json_data: dict
) -> Response:
    """
    Handles requests to the Google GenAI SDK.
    Delegates to the Google provider.
    """
    return handle_google_request(
        json_data_for_sdk,
        target_model,
        target_api_key,
        is_stream,
        original_request_json_data
    )


def handle_claude_code_sdk_request(
    json_data_for_sdk: dict,
    target_model: str,
    target_api_key: str | None,
    is_stream: bool,
    original_request_json_data: dict,
    claude_code_path: str | None = None,
    claude_code_oauth_token: str | None = None,
    max_output_tokens: int | None = None
) -> Response:
    """
    Handles requests to Claude Code.
    Delegates to the Claude Code provider.
    """
    return handle_claude_code_request(
        json_data_for_sdk,
        target_model,
        target_api_key,
        is_stream,
        original_request_json_data,
        claude_code_path,
        claude_code_oauth_token,
        max_output_tokens
    )


def handle_rest_api_request(
    method: str, 
    url: str, 
    headers: dict, 
    data_bytes: bytes, 
    is_stream: bool, 
    original_request_json_data: dict,
    provider: str = "openai"
) -> Response:
    """
    Handles requests to REST APIs by routing to the appropriate provider.
    Delegates to the specific provider based on the provider parameter.
    """
    # Route to the appropriate provider
    if provider == "google":
        return handle_google_request(
            method, url, headers, data_bytes, is_stream, original_request_json_data
        )
    elif provider == "ollama":
        return handle_ollama_request(
            method, url, headers, data_bytes, is_stream, original_request_json_data
        )
    else:
        # Default to OpenAI provider for OpenAI-compatible APIs
        return handle_openai_request(
            method, url, headers, data_bytes, is_stream, original_request_json_data
        )
