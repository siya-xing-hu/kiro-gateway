# -*- coding: utf-8 -*-

"""
Converters for transforming OpenAI format to Qoder API format.

Since Qoder API is expected to be OpenAI-compatible, this module primarily:
- Resolves model IDs/aliases
- Validates and sanitizes request data
- Handles any minor format differences
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.qoder.config import resolve_model_id
from kiro.qoder.models import ChatCompletionRequest, Tool


def build_qoder_payload(request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Builds the request payload for Qoder API.
    
    Since Qoder API is expected to be OpenAI-compatible, this primarily
    involves:
    - Resolving model ID aliases
    - Converting Pydantic models to dicts
    - Handling any format differences
    
    Args:
        request: ChatCompletionRequest from client
    
    Returns:
        Dictionary ready for JSON serialization to Qoder API
    
    Example:
        >>> request = ChatCompletionRequest(model="claude-sonnet", messages=[...])
        >>> payload = build_qoder_payload(request)
        >>> # payload is ready for httpx.post(json=payload)
    """
    # Resolve model ID (handle aliases)
    model_id = resolve_model_id(request.model)
    
    # Start with basic payload
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [],
        "stream": request.stream,
    }
    
    # Convert messages
    for msg in request.messages:
        msg_dict = msg.model_dump(exclude_none=True)
        # Ensure content is properly formatted
        if msg_dict.get("content") is None:
            msg_dict["content"] = ""
        payload["messages"].append(msg_dict)
    
    # Add optional parameters if provided
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    elif request.max_completion_tokens is not None:
        # Use max_completion_tokens as fallback for max_tokens
        payload["max_tokens"] = request.max_completion_tokens
    
    if request.stop is not None:
        payload["stop"] = request.stop
    
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    
    # Handle tools
    if request.tools:
        payload["tools"] = convert_tools_to_qoder_format(request.tools)
        
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
    
    logger.debug(f"Built Qoder payload: model={model_id}, messages={len(payload['messages'])}, stream={request.stream}")
    
    return payload


def convert_tools_to_qoder_format(tools: List[Tool]) -> List[Dict[str, Any]]:
    """
    Converts tools to Qoder API format.
    
    Handles both standard OpenAI format and flat (Cursor-style) format.
    
    Args:
        tools: List of Tool objects from request
    
    Returns:
        List of tool dictionaries in Qoder API format
    """
    qoder_tools = []
    
    for tool in tools:
        # Standard OpenAI format: {"type": "function", "function": {...}}
        if tool.function:
            tool_dict = {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description or f"Tool: {tool.function.name}",
                    "parameters": tool.function.parameters or {}
                }
            }
        # Flat format (Cursor-style): {"name": "...", "input_schema": {...}}
        elif tool.name:
            tool_dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.input_schema or {}
                }
            }
        else:
            # Unknown format, skip
            logger.warning(f"Skipping tool with unknown format: {tool}")
            continue
        
        qoder_tools.append(tool_dict)
    
    return qoder_tools


def extract_system_prompt(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """
    Extracts system prompt from messages.
    
    Some APIs prefer system prompts to be separate from messages.
    This function extracts the first system message and returns
    remaining messages.
    
    Args:
        messages: List of message dictionaries
    
    Returns:
        Tuple of (system_prompt, remaining_messages)
    
    Example:
        >>> messages = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}]
        >>> system, remaining = extract_system_prompt(messages)
        >>> system
        'You are helpful'
        >>> remaining
        [{'role': 'user', 'content': 'Hi'}]
    """
    system_prompt = ""
    remaining_messages = []
    found_system = False
    
    for msg in messages:
        if msg.get("role") == "system" and not found_system:
            # Extract first system message
            content = msg.get("content", "")
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                # Handle list content format
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                system_prompt = "\n".join(text_parts)
            found_system = True
        else:
            remaining_messages.append(msg)
    
    return system_prompt, remaining_messages


def validate_request(request: ChatCompletionRequest) -> Optional[str]:
    """
    Validates the chat completion request.
    
    Returns an error message if validation fails, None if valid.
    
    Args:
        request: ChatCompletionRequest to validate
    
    Returns:
        Error message string or None if valid
    """
    # Check for messages
    if not request.messages:
        return "Messages list cannot be empty"
    
    # Check for at least one user message
    has_user_message = any(msg.role == "user" for msg in request.messages)
    if not has_user_message:
        return "At least one user message is required"
    
    # Validate temperature range
    if request.temperature is not None and not (0 <= request.temperature <= 2):
        return "Temperature must be between 0 and 2"
    
    # Validate top_p range
    if request.top_p is not None and not (0 <= request.top_p <= 1):
        return "top_p must be between 0 and 1"
    
    # Validate max_tokens
    if request.max_tokens is not None and request.max_tokens <= 0:
        return "max_tokens must be positive"
    
    return None
