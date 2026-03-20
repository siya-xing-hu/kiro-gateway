# -*- coding: utf-8 -*-

"""
Streaming logic for converting Qoder API stream to OpenAI format.

Handles SSE (Server-Sent Events) parsing and conversion to OpenAI format.
Since Qoder API is expected to be OpenAI-compatible, this module primarily
handles SSE parsing and chunk formatting.
"""

import json
import time
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger

from kiro.qoder.config import resolve_model_id


def generate_completion_id() -> str:
    """
    Generates a unique completion ID.
    
    Returns:
        Unique string ID for the completion
    """
    import uuid
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


async def stream_qoder_to_openai(
    response: httpx.Response,
    model: str
) -> AsyncGenerator[str, None]:
    """
    Converts Qoder streaming response to OpenAI SSE format.
    
    Parses SSE events from Qoder API and converts them to OpenAI format.
    Since Qoder API is expected to be OpenAI-compatible, this primarily
    involves parsing SSE and ensuring proper formatting.
    
    Args:
        response: HTTP response with streaming data
        model: Model name to include in response
    
    Yields:
        Strings in SSE format: "data: {...}\\n\\n" or "data: [DONE]\\n\\n"
    
    Example:
        >>> async for chunk in stream_qoder_to_openai(response, "claude-sonnet"):
        ...     print(chunk)
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}
        
        data: [DONE]
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    resolved_model = resolve_model_id(model)
    
    first_chunk = True
    buffer = ""
    
    try:
        async for line in response.aiter_lines():
            # Skip empty lines
            if not line.strip():
                continue
            
            # Handle SSE format
            if line.startswith("data: "):
                data = line[6:]  # Remove "data: " prefix
                
                # Check for end of stream
                if data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                
                # Parse the JSON data
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse SSE data: {data[:100]}")
                    continue
                
                # Convert to OpenAI format if needed
                openai_chunk = convert_to_openai_chunk(
                    chunk_data,
                    completion_id,
                    created_time,
                    resolved_model,
                    first_chunk
                )
                first_chunk = False
                
                # Yield the formatted chunk
                chunk_str = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                yield chunk_str
            
            else:
                # Handle non-SSE format (raw JSON lines)
                try:
                    chunk_data = json.loads(line)
                    openai_chunk = convert_to_openai_chunk(
                        chunk_data,
                        completion_id,
                        created_time,
                        resolved_model,
                        first_chunk
                    )
                    first_chunk = False
                    chunk_str = f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n"
                    yield chunk_str
                except json.JSONDecodeError:
                    # Accumulate in buffer for multi-line JSON
                    buffer += line
                    continue
    
    except httpx.ReadError as e:
        logger.error(f"Stream read error: {e}")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Unexpected streaming error: {e}")
        yield "data: [DONE]\n\n"


def convert_to_openai_chunk(
    chunk_data: dict,
    completion_id: str,
    created_time: int,
    model: str,
    is_first: bool
) -> dict:
    """
    Converts a chunk from Qoder API to OpenAI format.
    
    Handles both OpenAI-compatible chunks and potential Qoder-specific formats.
    
    Args:
        chunk_data: Parsed JSON data from Qoder API
        completion_id: Unique completion ID
        created_time: Creation timestamp
        model: Model name
        is_first: Whether this is the first chunk
    
    Returns:
        Dictionary in OpenAI chunk format
    """
    # If already in OpenAI format, just update IDs
    if "choices" in chunk_data:
        chunk = chunk_data.copy()
        chunk["id"] = completion_id
        chunk["created"] = created_time
        chunk["model"] = model
        
        # Ensure first chunk has role
        if is_first and chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            if "role" not in delta:
                delta["role"] = "assistant"
        
        return chunk
    
    # Handle alternative formats
    # Format: {"content": "...", "finish_reason": null}
    if "content" in chunk_data or "text" in chunk_data:
        content = chunk_data.get("content") or chunk_data.get("text", "")
        finish_reason = chunk_data.get("finish_reason")
        
        delta = {"content": content}
        if is_first:
            delta["role"] = "assistant"
        
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
    
    # Handle tool calls in stream
    if "tool_calls" in chunk_data:
        delta = {"tool_calls": chunk_data["tool_calls"]}
        if is_first:
            delta["role"] = "assistant"
        
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": chunk_data.get("finish_reason")
            }]
        }
    
    # Handle usage information
    if "usage" in chunk_data:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": None
            }],
            "usage": chunk_data["usage"]
        }
    
    # Unknown format - log and return as-is with minimal wrapping
    logger.debug(f"Unknown chunk format: {list(chunk_data.keys())}")
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": chunk_data,
            "finish_reason": None
        }]
    }


async def collect_stream_response(
    response: httpx.Response,
    model: str
) -> dict:
    """
    Collects entire streaming response into a single OpenAI response.
    
    Used for non-streaming requests where the API returns streaming format.
    
    Args:
        response: HTTP response with streaming data
        model: Model name
    
    Returns:
        Complete response dictionary in OpenAI format
    """
    completion_id = generate_completion_id()
    created_time = int(time.time())
    resolved_model = resolve_model_id(model)
    
    full_content = ""
    tool_calls = []
    finish_reason = None
    usage = None
    
    try:
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            
            if line.startswith("data: "):
                data = line[6:]
                
                if data == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                
                # Extract content from chunk
                if "choices" in chunk_data:
                    choices = chunk_data["choices"]
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content += content
                        
                        # Collect tool calls
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])
                        
                        # Get finish reason
                        if choices[0].get("finish_reason"):
                            finish_reason = choices[0]["finish_reason"]
                
                # Extract usage
                if "usage" in chunk_data:
                    usage = chunk_data["usage"]
    
    except Exception as e:
        logger.error(f"Error collecting stream response: {e}")
    
    # Build final response
    message: dict = {"role": "assistant", "content": full_content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": resolved_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
