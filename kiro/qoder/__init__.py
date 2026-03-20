# -*- coding: utf-8 -*-

"""
Qoder CLI Proxy Module.

This module provides a proxy gateway for Qoder CLI,
exposing an OpenAI-compatible interface for seamless integration
with tools that support OpenAI API.

Key Components:
- config: Configuration management
- models: Pydantic models for request/response validation
- cli_client: CLI client that executes qodercli commands
- converters: Request format conversion
- routes: FastAPI route handlers

Usage:
    The module is automatically integrated into the main FastAPI application.
    Access the API endpoints at:
    - GET  /qoder/           - Health check
    - GET  /qoder/health     - Detailed health status
    - GET  /qoder/v1/models  - List available models
    - POST /qoder/v1/chat/completions - Chat completions
"""

from kiro.qoder.config import (
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
    QODER_PROXY_API_KEY,
    resolve_model_id,
)
from kiro.qoder.cli_client import QoderCliClient, get_cli_client
from kiro.qoder.routes import router

__all__ = [
    # Config
    "APP_TITLE",
    "APP_VERSION",
    "APP_DESCRIPTION",
    "QODER_PROXY_API_KEY",
    "resolve_model_id",
    # CLI Client
    "QoderCliClient",
    "get_cli_client",
    # Routes
    "router",
]
