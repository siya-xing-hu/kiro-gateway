# -*- coding: utf-8 -*-

"""
HTTP client for Qoder API with retry logic support.

Handles:
- 401: authentication errors
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
from typing import Optional, Dict, Any

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.qoder.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    STREAMING_READ_TIMEOUT,
    FIRST_TOKEN_MAX_RETRIES,
    get_qoder_chat_url,
)
from kiro.qoder.auth import QoderAuthManager


class QoderHttpClient:
    """
    HTTP client for Qoder API with retry logic support.
    
    Automatically handles errors and retries requests:
    - 401: authentication errors (try to reload token)
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - Timeouts: waits with exponential backoff
    
    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)
    
    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.
    
    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)
    
    Example:
        >>> # Per-request client (legacy mode)
        >>> client = QoderHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)
        
        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = QoderHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """
    
    def __init__(
        self,
        auth_manager: QoderAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.
        
        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client
    
    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.
        
        If a shared client was provided at initialization, it is returned as-is.
        Otherwise, creates a new client with appropriate timeout configuration.
        
        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)
        
        Returns:
            Active HTTP client
        """
        # If using shared client, return it directly
        if self._shared_client is not None:
            return self._shared_client
        
        # Create new client if needed (per-request mode)
        if self.client is None or self.client.is_closed:
            if stream:
                # For streaming:
                # - connect: 30 sec (TCP connection)
                # - read: STREAMING_READ_TIMEOUT (300 sec) - model may "think" between chunks
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                )
                logger.debug(f"Creating streaming HTTP client (read_timeout={STREAMING_READ_TIMEOUT}s)")
            else:
                # For regular requests: single timeout of 300 sec
                timeout_config = httpx.Timeout(timeout=300.0)
                logger.debug("Creating non-streaming HTTP client (timeout=300s)")
            
            self.client = httpx.AsyncClient(timeout=timeout_config, follow_redirects=True)
        return self.client
    
    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.
        
        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.
        """
        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return
        
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing HTTP client: {e}")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Returns headers for Qoder API requests.
        
        Returns:
            Dictionary of headers including Authorization
        
        Raises:
            ValueError: If no token is configured
        """
        return {
            "Authorization": self.auth_manager.get_auth_header(),
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self._streaming_mode else "application/json",
        }
    
    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Dict[str, Any],
        stream: bool = False
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.
        
        Automatically handles various error types:
        - 401: tries to reload token and retries
        - 429: waits with exponential backoff
        - 5xx: waits with exponential backoff
        - Timeouts: waits with exponential backoff
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Request body (JSON)
            stream: Use streaming (default False)
        
        Returns:
            httpx.Response with successful response
        
        Raises:
            HTTPException: On failure after all attempts (502/504)
        """
        # Store streaming mode for headers
        self._streaming_mode = stream
        
        # Determine the number of retry attempts
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        
        client = await self._get_client(stream=stream)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Get headers with current token
                headers = self._get_headers()
                
                if stream:
                    # For streaming, prevent CLOSE_WAIT connection leak
                    headers["Connection"] = "close"
                    req = client.build_request(method, url, json=json_data, headers=headers)
                    logger.debug("Sending request to Qoder API...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug("Sending request to Qoder API...")
                    response = await client.request(method, url, json=json_data, headers=headers)
                
                # Check status
                if response.status_code == 200:
                    return response
                
                # 401 - authentication error, try to reload token
                if response.status_code == 401:
                    logger.warning(f"Received 401, reloading token (attempt {attempt + 1}/{max_retries})")
                    if self.auth_manager.reload_token():
                        continue
                    # If reload failed, raise error
                    raise HTTPException(
                        status_code=401,
                        detail="Qoder API authentication failed. Please check your Personal Access Token."
                    )
                
                # 403 - forbidden, might be token issue
                if response.status_code == 403:
                    logger.warning(f"Received 403 from Qoder API (attempt {attempt + 1}/{max_retries})")
                    # Try to reload token
                    self.auth_manager.reload_token()
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                
                # 429 - rate limit, wait and retry
                if response.status_code == 429:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Received 429, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # 5xx - server error, wait and retry
                if 500 <= response.status_code < 600:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Received {response.status_code}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                
                # Other errors - return as is
                return response
                
            except httpx.TimeoutException as e:
                last_error = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request timeout, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            
            except httpx.RequestError as e:
                last_error = e
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request error: {e}, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
        
        # All attempts exhausted
        if last_error:
            raise HTTPException(
                status_code=504 if stream else 502,
                detail=f"Request to Qoder API failed after {max_retries} attempts: {last_error}"
            )
        else:
            raise HTTPException(
                status_code=504 if stream else 502,
                detail=f"Request to Qoder API failed after {max_retries} attempts"
            )
    
    async def __aenter__(self) -> "QoderHttpClient":
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()
