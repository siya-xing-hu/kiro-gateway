# -*- coding: utf-8 -*-

"""
Qoder CLI Proxy Configuration.

Centralized storage for all Qoder-specific settings, constants, and mappings.
Loads environment variables and provides typed access to them.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# ==================================================================================================
# Server Settings
# ==================================================================================================

# Server host (default: same as Kiro)
DEFAULT_SERVER_HOST: str = "0.0.0.0"
SERVER_HOST: str = os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST)

# Server port (default: 8000)
DEFAULT_SERVER_PORT: int = 8000
SERVER_PORT: int = int(os.getenv("SERVER_PORT", str(DEFAULT_SERVER_PORT)))


# ==================================================================================================
# Qoder API Configuration
# ==================================================================================================

# Qoder API base URL
# Default: https://api.qoder.com (to be confirmed)
QODER_API_BASE_URL: str = os.getenv("QODER_API_BASE_URL", "https://api.qoder.com")

# Qoder API version (if needed)
QODER_API_VERSION: str = os.getenv("QODER_API_VERSION", "v1")


# ==================================================================================================
# Authentication Settings
# ==================================================================================================

# Personal Access Token for Qoder API
# Can be obtained from: https://qoder.com/account/integrations
QODER_PERSONAL_ACCESS_TOKEN: str = os.getenv("QODER_PERSONAL_ACCESS_TOKEN", "")

# Path to Qoder CLI config file (alternative to environment variable)
# Default location: ~/.qoder.json
QODER_CONFIG_FILE: str = os.getenv("QODER_CONFIG_FILE", "")

# Proxy API key for accessing this gateway
# Clients must pass this in Authorization header
QODER_PROXY_API_KEY: str = os.getenv("QODER_PROXY_API_KEY", "my-qoder-secret-password-123")


# ==================================================================================================
# Retry Configuration
# ==================================================================================================

# Maximum number of retry attempts on errors
MAX_RETRIES: int = 3

# Base delay between attempts (seconds)
# Uses exponential backoff: delay * (2 ** attempt)
BASE_RETRY_DELAY: float = 1.0


# ==================================================================================================
# Timeout Settings
# ==================================================================================================

# Timeout for waiting for the first token from the model (in seconds)
FIRST_TOKEN_TIMEOUT: float = float(os.getenv("QODER_FIRST_TOKEN_TIMEOUT", "30"))

# Read timeout for streaming responses (in seconds)
STREAMING_READ_TIMEOUT: float = float(os.getenv("QODER_STREAMING_READ_TIMEOUT", "300"))

# Maximum number of attempts on first token timeout
FIRST_TOKEN_MAX_RETRIES: int = int(os.getenv("QODER_FIRST_TOKEN_MAX_RETRIES", "3"))


# ==================================================================================================
# Logging Settings
# ==================================================================================================

# Log level for the application
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


# ==================================================================================================
# Model Configuration
# ==================================================================================================

# Qoder uses tiered models instead of specific model names
# See: https://docs.qoder.com/cli/model
#
# Tiered Models (Default):
# - lite: Free - Simple Q&A, lightweight tasks
# - efficient: Low - Everyday coding, code completion
# - auto: Standard - Complex tasks, multi-step reasoning (default)
# - performance: High - Challenging engineering problems, large codebases
# - ultimate: Highest - Maximum performance, best possible results

QODER_DEFAULT_MODELS: List[Dict[str, str]] = [
    {"id": "lite", "owned_by": "qoder", "description": "Free tier - Simple Q&A, lightweight tasks"},
    {"id": "efficient", "owned_by": "qoder", "description": "Low cost - Everyday coding, code completion"},
    {"id": "auto", "owned_by": "qoder", "description": "Standard - Complex tasks, multi-step reasoning (default)"},
    {"id": "performance", "owned_by": "qoder", "description": "High cost - Challenging engineering problems, large codebases"},
    {"id": "ultimate", "owned_by": "qoder", "description": "Highest cost - Maximum performance, best possible results"},
]

# Model aliases - map common names to Qoder tiers
# Users can use familiar model names, which will be mapped to appropriate tiers
QODER_MODEL_ALIASES: Dict[str, str] = {
    # Map to auto tier (default, standard)
    "claude-sonnet": "auto",
    "claude-sonnet-4": "auto",
    "claude-3.5-sonnet": "auto",
    "gpt-4": "auto",
    "gpt-4o": "auto",
    # Map to performance tier
    "claude-opus": "performance",
    "claude-opus-4": "performance",
    "claude-3.5-opus": "performance",
    # Map to ultimate tier
    "claude-opus-4.5": "ultimate",
    # Map to efficient tier
    "claude-haiku": "efficient",
    "claude-3.5-haiku": "efficient",
    "gpt-4o-mini": "efficient",
    # Map to lite tier
    "gpt-3.5-turbo": "lite",
}


# ==================================================================================================
# Application Version
# ==================================================================================================

APP_VERSION: str = "1.0.0"
APP_TITLE: str = "Qoder CLI Gateway"
APP_DESCRIPTION: str = "Proxy gateway for Qoder CLI API. OpenAI compatible. Integrated with Kiro Gateway."


# ==================================================================================================
# Helper Functions
# ==================================================================================================

def get_qoder_api_url() -> str:
    """
    Return the full Qoder API URL.
    
    Returns:
        Full API URL including version if specified
    """
    base_url = QODER_API_BASE_URL.rstrip("/")
    if QODER_API_VERSION:
        return f"{base_url}/{QODER_API_VERSION}"
    return base_url


def get_qoder_chat_url() -> str:
    """
    Return the Qoder chat completions URL.
    
    Returns:
        Full URL for chat completions endpoint
    """
    return f"{get_qoder_api_url()}/chat/completions"


def get_qoder_models_url() -> str:
    """
    Return the Qoder models list URL.
    
    Returns:
        Full URL for models endpoint
    """
    return f"{get_qoder_api_url()}/models"


def load_token_from_config_file(file_path: Optional[str] = None) -> Optional[str]:
    """
    Load Personal Access Token from Qoder CLI config file.
    
    The config file is typically located at ~/.qoder.json and may contain
    the token in various formats depending on Qoder CLI version.
    
    Args:
        file_path: Path to config file (default: ~/.qoder.json)
    
    Returns:
        Token string if found, None otherwise
    """
    if file_path:
        config_path = Path(file_path).expanduser()
    else:
        config_path = Path.home() / ".qoder.json"
    
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Try various possible token field names
        token_fields = [
            "personalAccessToken",
            "personal_access_token",
            "token",
            "accessToken",
            "access_token",
        ]
        
        for field in token_fields:
            if field in config and config[field]:
                return config[field]
        
        # Check nested structures
        if "auth" in config and isinstance(config["auth"], dict):
            for field in token_fields:
                if field in config["auth"] and config["auth"][field]:
                    return config["auth"][field]
        
        return None
    
    except (json.JSONDecodeError, IOError) as e:
        from loguru import logger
        logger.warning(f"Failed to load Qoder config from {config_path}: {e}")
        return None


def resolve_model_id(model_name: str) -> str:
    """
    Resolve model name to actual model ID.
    
    Checks aliases first, then returns the original name if no alias found.
    
    Args:
        model_name: Model name from client request
    
    Returns:
        Resolved model ID
    """
    return QODER_MODEL_ALIASES.get(model_name, model_name)
