"""
Unit tests for Google Gemini client integration and graceful fallback.
"""

import os
import pytest
from src.utils.gemini_client import get_gemini_api_key, is_gemini_configured, call_gemini_llm


def test_gemini_client_unconfigured_fallback():
    # Ensure without API key it returns None rather than raising an unhandled exception
    res = call_gemini_llm("Explain overdraft fee regulations")
    # If no key is set, returns None gracefully
    if not is_gemini_configured():
        assert res is None


def test_gemini_api_key_detection(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-12345")
    assert get_gemini_api_key() == "test-gemini-key-12345"
    assert is_gemini_configured() is True

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-67890")
    assert get_gemini_api_key() == "test-google-key-67890"
    assert is_gemini_configured() is True

