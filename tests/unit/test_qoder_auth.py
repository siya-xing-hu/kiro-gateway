# -*- coding: utf-8 -*-

"""
Tests for Qoder CLI Proxy authentication module.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kiro.qoder.auth import QoderAuthManager, AuthSource
from kiro.qoder.config import load_token_from_config_file


class TestQoderAuthManager:
    """Tests for QoderAuthManager class."""
    
    def test_init_with_direct_token(self):
        """Test initialization with directly provided token."""
        auth = QoderAuthManager(token="test-token-123")
        
        assert auth.is_configured() is True
        assert auth.get_token() == "test-token-123"
        assert auth.auth_source == AuthSource.DIRECT
    
    def test_init_with_env_variable(self):
        """Test initialization from environment variable."""
        # Clear any existing token and set new one
        with patch.dict(os.environ, {"QODER_PERSONAL_ACCESS_TOKEN": "env-token-456"}, clear=True):
            # Clear any cached imports
            import importlib
            import kiro.qoder.config
            importlib.reload(kiro.qoder.config)
            import kiro.qoder.auth
            importlib.reload(kiro.qoder.auth)
            from kiro.qoder.auth import QoderAuthManager, AuthSource
            
            auth = QoderAuthManager()
            
            assert auth.is_configured() is True
            assert auth.get_token() == "env-token-456"
            assert auth.auth_source.value == "environment"  # Compare string value
    
    def test_init_without_token(self):
        """Test initialization without any token source."""
        # Clear any environment variables and config file
        with patch.dict(os.environ, {}, clear=True):
            # Remove QODER_PERSONAL_ACCESS_TOKEN if present
            os.environ.pop("QODER_PERSONAL_ACCESS_TOKEN", None)
            
            # Reload modules to pick up cleared env
            import importlib
            import kiro.qoder.config
            importlib.reload(kiro.qoder.config)
            import kiro.qoder.auth
            importlib.reload(kiro.qoder.auth)
            from kiro.qoder.auth import QoderAuthManager
            
            # Use a non-existent config file to ensure no token is loaded
            auth = QoderAuthManager(config_file="/non/existent/path/config.json")
            
            assert auth.is_configured() is False
    
    def test_get_token_raises_when_not_configured(self):
        """Test that get_token raises ValueError when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("QODER_PERSONAL_ACCESS_TOKEN", None)
            
            # Reload modules to pick up cleared env
            import importlib
            import kiro.qoder.config
            importlib.reload(kiro.qoder.config)
            import kiro.qoder.auth
            importlib.reload(kiro.qoder.auth)
            from kiro.qoder.auth import QoderAuthManager
            
            # Use a non-existent config file to ensure no token is loaded
            auth = QoderAuthManager(config_file="/non/existent/path/config.json")
            
            with pytest.raises(ValueError) as exc_info:
                auth.get_token()
            
            assert "Personal Access Token not configured" in str(exc_info.value)
    
    def test_get_auth_header(self):
        """Test Authorization header generation."""
        auth = QoderAuthManager(token="test-token")
        
        assert auth.get_auth_header() == "Bearer test-token"
    
    def test_reload_token_from_file(self):
        """Test token reload from config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"personalAccessToken": "file-token-789"}, f)
            config_path = f.name
        
        try:
            # First create without token (use non-existent config)
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("QODER_PERSONAL_ACCESS_TOKEN", None)
                
                import importlib
                import kiro.qoder.config
                importlib.reload(kiro.qoder.config)
                import kiro.qoder.auth
                importlib.reload(kiro.qoder.auth)
                from kiro.qoder.auth import QoderAuthManager
                
                auth = QoderAuthManager(config_file="/non/existent/path/config.json")
                assert auth.is_configured() is False
                
                # Now set the config file and reload
                auth._config_file = config_path
                result = auth.reload_token()
                assert result is True
                assert auth.get_token() == "file-token-789"
        finally:
            os.unlink(config_path)


class TestLoadTokenFromConfigFile:
    """Tests for load_token_from_config_file function."""
    
    def test_load_token_with_personalAccessToken(self):
        """Test loading token with personalAccessToken field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"personalAccessToken": "test-token-123"}, f)
            config_path = f.name
        
        try:
            token = load_token_from_config_file(config_path)
            assert token == "test-token-123"
        finally:
            os.unlink(config_path)
    
    def test_load_token_with_token_field(self):
        """Test loading token with token field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"token": "test-token-456"}, f)
            config_path = f.name
        
        try:
            token = load_token_from_config_file(config_path)
            assert token == "test-token-456"
        finally:
            os.unlink(config_path)
    
    def test_load_token_with_nested_auth(self):
        """Test loading token from nested auth object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"auth": {"personalAccessToken": "nested-token-789"}}, f)
            config_path = f.name
        
        try:
            token = load_token_from_config_file(config_path)
            assert token == "nested-token-789"
        finally:
            os.unlink(config_path)
    
    def test_load_token_file_not_found(self):
        """Test loading token from non-existent file."""
        token = load_token_from_config_file("/non/existent/path.json")
        assert token is None
    
    def test_load_token_invalid_json(self):
        """Test loading token from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json")
            config_path = f.name
        
        try:
            token = load_token_from_config_file(config_path)
            assert token is None
        finally:
            os.unlink(config_path)
    
    def test_load_token_no_token_field(self):
        """Test loading token from file without token fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"otherField": "value"}, f)
            config_path = f.name
        
        try:
            token = load_token_from_config_file(config_path)
            assert token is None
        finally:
            os.unlink(config_path)
