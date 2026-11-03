"""
src/utils/gemini_client.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Google Gemini LLM Integration Client.

Provides unified interface for Google Gemini models with automatic candidate model resolution,
REST API fallback, and offline deterministic logic.
"""

import os
import sys
from pathlib import Path

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from typing import Optional, Dict, Any, List
from config.settings import LLM_MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini or Google API key from environment variables."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()
    return None


def is_gemini_configured() -> bool:
    """Return True if a valid Gemini API key is found in environment."""
    return get_gemini_api_key() is not None


def call_gemini_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    response_json: bool = False,
) -> Optional[str]:
    """Generate completion using Google Gemini API.

    Tries configured model first, followed by resilient available aliases:
      ['gemini-flash-lite-latest', 'gemini-flash-latest']

    Parameters
    ----------
    prompt : str
        Main user query or evaluation prompt.
    system_instruction : str, optional
        System-level directive / persona instructions.
    model : str, optional
        Gemini model identifier (defaults to LLM_MODEL_NAME or 'gemini-flash-lite-latest').
    temperature : float
        Sampling temperature (default 0.2 for analytical precision).
    response_json : bool
        If True, requests structured JSON output.

    Returns
    -------
    str or None
        Generated text / JSON string, or None if unavailable.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.debug("GEMINI_API_KEY / GOOGLE_API_KEY not set. Gemini generation skipped.")
        return None

    configured_model = model or LLM_MODEL_NAME or "gemini-flash-lite-latest"
    candidate_models: List[str] = [configured_model]
    for fallback in ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    import requests

    for m in candidate_models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            payload: Dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            }
            if system_instruction:
                payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
            if response_json:
                payload["generationConfig"]["responseMimeType"] = "application/json"

            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                        if text:
                            logger.info("Successfully generated response using model '%s'", m)
                            return text
            else:
                err_msg = resp.json().get("error", {}).get("message", "")
                logger.debug("Model '%s' returned HTTP %d: %s", m, resp.status_code, err_msg)
        except Exception as err:
            logger.debug("Model '%s' request failed (%s). Trying next candidate...", m, err)

    return None
