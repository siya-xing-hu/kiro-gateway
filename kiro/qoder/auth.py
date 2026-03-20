# -*- coding: utf-8 -*-

"""
Authentication manager for Qoder CLI API.

Manages Personal Access Token authentication:
- Loading token from environment variables
- Loading token from config file (~/.qoder.json)
- Token validation
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from loguru import logger

from kiro.qoder.config import (
    QODER_PERSONAL_ACCESS_TOKEN,
    QODER_CONFIG_FILE,
    load_token_from_config_file,
)


class AuthSource(Enum):
    """
    Source of the authentication token.
    
    ENVIRONMENT: Token loaded from environment variable
    CONFIG_FILE: Token loaded from ~/.qoder.json
    DIRECT: Token passed directly to constructor
    """
    ENVIRONMENT = "environment"
    CONFIG_FILE = "config_file"
    DIRECT = "direct"


class QoderAuthManager:
    """
    Manages authentication for Qoder CLI API.
    
    Supports Personal Access Token authentication from multiple sources:
    1. Direct token parameter (highest priority)
    2. Environment variable QODER_PERSONAL_ACCESS_TOKEN
    3. Config file ~/.qoder.json
    
    The Personal Access Token is obtained from:
    https://qoder.com/account/integrations
    
    Attributes:
        token: The Personal Access Token
        auth_source: Where the token was loaded from
    
    Example:
        >>> # From environment variable
        >>> auth = QoderAuthManager()
        >>> token = auth.get_token()
        
        >>> # From direct parameter
        >>> auth = QoderAuthManager(token="your-token-here")
        >>> token = auth.get_token()
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        """
        Initializes the authentication manager.
        
        Token loading priority:
        1. Direct token parameter
        2. Environment variable QODER_PERSONAL_ACCESS_TOKEN
        3. Config file (QODER_CONFIG_FILE or ~/.qoder.json)
        
        Args:
            token: Personal Access Token (optional, will try to load from other sources if not provided)
            config_file: Path to Qoder config file (optional, defaults to ~/.qoder.json)
        """
        self._token: Optional[str] = None
        self._auth_source: Optional[AuthSource] = None
        self._config_file = config_file or QODER_CONFIG_FILE
        
        # Load token from sources
        if token:
            self._token = token
            self._auth_source = AuthSource.DIRECT
            logger.info("Qoder auth: Using directly provided token")
        elif QODER_PERSONAL_ACCESS_TOKEN:
            self._token = QODER_PERSONAL_ACCESS_TOKEN
            self._auth_source = AuthSource.ENVIRONMENT
            logger.info("Qoder auth: Using token from environment variable")
        else:
            # Try to load from config file
            loaded_token = load_token_from_config_file(self._config_file)
            if loaded_token:
                self._token = loaded_token
                self._auth_source = AuthSource.CONFIG_FILE
                config_path = self._config_file or "~/.qoder.json"
                logger.info(f"Qoder auth: Using token from config file ({config_path})")
        
        if not self._token:
            logger.warning(
                "Qoder auth: No Personal Access Token configured. "
                "Set QODER_PERSONAL_ACCESS_TOKEN environment variable or "
                "create ~/.qoder.json config file."
            )
    
    def get_token(self) -> str:
        """
        Returns the Personal Access Token.
        
        Returns:
            The Personal Access Token string
        
        Raises:
            ValueError: If no token is configured
        """
        if not self._token:
            raise ValueError(
                "Qoder Personal Access Token not configured. "
                "Please set QODER_PERSONAL_ACCESS_TOKEN environment variable "
                "or create ~/.qoder.json config file with your token. "
                "Get your token from: https://qoder.com/account/integrations"
            )
        return self._token
    
    def is_configured(self) -> bool:
        """
        Check if authentication is configured.
        
        Returns:
            True if a token is available, False otherwise
        """
        return bool(self._token)
    
    @property
    def auth_source(self) -> Optional[AuthSource]:
        """
        Returns the source of the authentication token.
        
        Returns:
            AuthSource enum value or None if not configured
        """
        return self._auth_source
    
    def reload_token(self) -> bool:
        """
        Reload token from configuration sources.
        
        Useful when the config file might have been updated externally.
        
        Returns:
            True if token was successfully reloaded, False otherwise
        """
        # Try to load from config file
        loaded_token = load_token_from_config_file(self._config_file)
        if loaded_token:
            self._token = loaded_token
            self._auth_source = AuthSource.CONFIG_FILE
            logger.info("Qoder auth: Token reloaded from config file")
            return True
        
        # Fall back to environment variable
        if QODER_PERSONAL_ACCESS_TOKEN:
            self._token = QODER_PERSONAL_ACCESS_TOKEN
            self._auth_source = AuthSource.ENVIRONMENT
            logger.info("Qoder auth: Token reloaded from environment variable")
            return True
        
        logger.warning("Qoder auth: Failed to reload token - no token found in config sources")
        return False
    
    def get_auth_header(self) -> str:
        """
        Returns the Authorization header value.
        
        Returns:
            Authorization header value in format "Bearer {token}"
        
        Raises:
            ValueError: If no token is configured
        """
        return f"Bearer {self.get_token()}"
