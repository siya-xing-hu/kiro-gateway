# -*- coding: utf-8 -*-

"""
Tests for Qoder CLI Proxy converters module.
"""

import pytest

from kiro.qoder.models import ChatCompletionRequest, ChatMessage, Tool, ToolFunction
from kiro.qoder.converters import (
    build_qoder_payload,
    convert_tools_to_qoder_format,
    extract_system_prompt,
    validate_request,
)


class TestBuildQoderPayload:
    """Tests for build_qoder_payload function."""
    
    def test_basic_request(self):
        """Test building payload from basic request."""
        request = ChatCompletionRequest(
            model="auto",  # Use Qoder tier name
            messages=[
                ChatMessage(role="user", content="Hello")
            ]
        )
        
        payload = build_qoder_payload(request)
        
        assert payload["model"] == "auto"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["stream"] is False
    
    def test_request_with_model_alias(self):
        """Test that model aliases are resolved to Qoder tiers."""
        request = ChatCompletionRequest(
            model="claude-sonnet",  # This should be resolved to 'auto' tier
            messages=[
                ChatMessage(role="user", content="Hello")
            ]
        )
        
        payload = build_qoder_payload(request)
        
        # Model should be resolved via alias to 'auto' tier
        assert payload["model"] == "auto"
    
    def test_request_with_temperature(self):
        """Test request with temperature parameter."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7
        )
        
        payload = build_qoder_payload(request)
        
        assert payload["temperature"] == 0.7
    
    def test_request_with_max_tokens(self):
        """Test request with max_tokens parameter."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=1000
        )
        
        payload = build_qoder_payload(request)
        
        assert payload["max_tokens"] == 1000
    
    def test_request_with_streaming(self):
        """Test request with streaming enabled."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            stream=True
        )
        
        payload = build_qoder_payload(request)
        
        assert payload["stream"] is True
    
    def test_request_with_tools(self):
        """Test request with tools."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            tools=[
                Tool(
                    type="function",
                    function=ToolFunction(
                        name="test_tool",
                        description="A test tool",
                        parameters={"type": "object", "properties": {}}
                    )
                )
            ]
        )
        
        payload = build_qoder_payload(request)
        
        assert "tools" in payload
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["function"]["name"] == "test_tool"


class TestConvertToolsToQoderFormat:
    """Tests for convert_tools_to_qoder_format function."""
    
    def test_standard_openai_format(self):
        """Test conversion of standard OpenAI format tools."""
        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather info",
                    parameters={"type": "object", "properties": {"location": {"type": "string"}}}
                )
            )
        ]
        
        result = convert_tools_to_qoder_format(tools)
        
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather info"
    
    def test_flat_cursor_format(self):
        """Test conversion of flat (Cursor-style) format tools."""
        tools = [
            Tool(
                name="bash",
                description="Run bash commands",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}}}
            )
        ]
        
        result = convert_tools_to_qoder_format(tools)
        
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert result[0]["function"]["description"] == "Run bash commands"
    
    def test_tool_without_description(self):
        """Test tool without description gets placeholder."""
        tools = [
            Tool(
                type="function",
                function=ToolFunction(name="test_tool")
            )
        ]
        
        result = convert_tools_to_qoder_format(tools)
        
        assert result[0]["function"]["description"] == "Tool: test_tool"


class TestExtractSystemPrompt:
    """Tests for extract_system_prompt function."""
    
    def test_extract_single_system_message(self):
        """Test extracting a single system message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        system, remaining = extract_system_prompt(messages)
        
        assert system == "You are helpful"
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"
    
    def test_extract_no_system_message(self):
        """Test when there is no system message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        system, remaining = extract_system_prompt(messages)
        
        assert system == ""
        assert len(remaining) == 2
    
    def test_extract_multiple_system_messages(self):
        """Test that only first system message is extracted."""
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "system", "content": "Second system"},
            {"role": "user", "content": "Hello"}
        ]
        
        system, remaining = extract_system_prompt(messages)
        
        assert system == "First system"
        assert len(remaining) == 2  # Second system + user


class TestValidateRequest:
    """Tests for validate_request function."""
    
    def test_valid_request(self):
        """Test validation of valid request."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")]
        )
        
        error = validate_request(request)
        
        assert error is None
    
    def test_empty_messages(self):
        """Test validation fails with empty messages."""
        # This would fail at Pydantic validation level
        # So we test with at least one message but no user message
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="assistant", content="Hello")]
        )
        
        error = validate_request(request)
        
        assert error == "At least one user message is required"
    
    def test_invalid_temperature(self):
        """Test validation fails with invalid temperature."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=3.0  # Invalid: > 2
        )
        
        error = validate_request(request)
        
        assert error == "Temperature must be between 0 and 2"
    
    def test_invalid_top_p(self):
        """Test validation fails with invalid top_p."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            top_p=1.5  # Invalid: > 1
        )
        
        error = validate_request(request)
        
        assert error == "top_p must be between 0 and 1"
    
    def test_invalid_max_tokens(self):
        """Test validation fails with invalid max_tokens."""
        request = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=-100  # Invalid: negative
        )
        
        error = validate_request(request)
        
        assert error == "max_tokens must be positive"
