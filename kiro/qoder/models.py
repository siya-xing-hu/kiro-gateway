# -*- coding: utf-8 -*-

"""
Pydantic models for Qoder CLI Proxy API.

Reuses OpenAI-compatible models from kiro.models_openai for consistency.
The Qoder API is expected to be OpenAI-compatible, so we can use the same models.
"""

# Re-export all models from OpenAI module
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatMessage,
    ToolFunction,
    Tool,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse,
    ChatCompletionChunkDelta,
    ChatCompletionChunkChoice,
    ChatCompletionChunk,
)

__all__ = [
    # Models endpoint
    "OpenAIModel",
    "ModelList",
    # Chat completions request
    "ChatMessage",
    "ToolFunction",
    "Tool",
    "ChatCompletionRequest",
    # Chat completions response
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionResponse",
    # Streaming
    "ChatCompletionChunkDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunk",
]
