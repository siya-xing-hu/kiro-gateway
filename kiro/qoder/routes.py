# -*- coding: utf-8 -*-

"""
FastAPI routes for Qoder CLI Proxy.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions

This implementation uses qodercli command directly instead of HTTP API.
"""

import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.qoder.config import (
    QODER_PROXY_API_KEY,
    QODER_DEFAULT_MODELS,
    APP_VERSION,
)
from kiro.qoder.models import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.qoder.cli_client import get_cli_client, QoderCliClient
from kiro.qoder.converters import validate_request


# --- Security scheme ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.
    
    Expects format: "Bearer {QODER_PROXY_API_KEY}"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header or auth_header != f"Bearer {QODER_PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key for Qoder proxy.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Qoder CLI Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health():
    """
    Detailed health check.
    
    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION
    }


@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    """
    Return list of available models.
    
    Returns the default model list configured for Qoder.
    
    Args:
        request: FastAPI Request for accessing app.state
    
    Returns:
        ModelList with available models
    """
    logger.info("Request to /qoder/v1/models")
    
    # Build OpenAI-compatible model list
    openai_models = [
        OpenAIModel(
            id=model["id"],
            owned_by=model.get("owned_by", "qoder"),
            description=model.get("description")
        )
        for model in QODER_DEFAULT_MODELS
    ]
    
    return ModelList(data=openai_models)


@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.
    
    Accepts requests in OpenAI format and executes qodercli command.
    Supports streaming and non-streaming modes.
    
    Args:
        request: FastAPI Request
        request_data: Request in OpenAI ChatCompletionRequest format
    
    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or CLI errors
    """
    logger.info(f"Request to /qoder/v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    
    # Get CLI client
    cli_client = get_cli_client()
    
    # Check if qodercli is available
    if not cli_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="qodercli command not found. Please install Qoder CLI: https://docs.qoder.com/cli"
        )
    
    # Validate request
    validation_error = validate_request(request_data)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    # Convert messages to format expected by CLI
    messages = []
    for msg in request_data.messages:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    try:
        if request_data.stream:
            # Streaming mode
            async def stream_wrapper():
                """Generate SSE stream from CLI output."""
                try:
                    # Send initial chunk with role
                    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    created = int(time.time())
                    
                    initial_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request_data.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    
                    # Stream content from CLI
                    content_buffer = ""
                    async for content in cli_client.chat_completion_stream(
                        messages=messages,
                        model=request_data.model,
                    ):
                        content_buffer += content
                        
                        # Yield chunk
                        chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request_data.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    
                    # Send final chunk
                    final_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request_data.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    logger.info("HTTP 200 - POST /qoder/v1/chat/completions (streaming) - completed")
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    raise
            
            return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
        
        else:
            # Non-streaming mode
            response = await cli_client.chat_completion(
                messages=messages,
                model=request_data.model,
                stream=False,
                temperature=request_data.temperature,
                max_tokens=request_data.max_tokens,
            )
            
            logger.info("HTTP 200 - POST /qoder/v1/chat/completions (non-streaming) - completed")
            
            return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
