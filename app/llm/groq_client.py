"""
Groq API client with multi-key rotation and model fallback.

This module ONLY manages the Groq API connection and raw completions.
No retrieval, no searching — just authenticated LLM calls.

Usage:
    client = GroqAPIClient()
    raw_json_str = client.call(messages)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import groq
from groq import Groq

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqAPIClient:
    """
    Low-level Groq API wrapper.

    Responsibilities:
    - Multi-key round-robin rotation on rate limits (429).
    - Ordered model fallback on decommissioned/unavailable models.
    - Zero SDK-level retries (instant failover, no blocking sleeps).
    - Returns raw JSON string — callers parse and validate.
    """

    def __init__(self) -> None:
        self.api_keys: List[str] = settings.get_api_keys()
        self.models: List[str] = settings.get_fallback_models()
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.timeout = settings.timeout_seconds
        self._key_index = 0

        if not self.api_keys:
            logger.warning("No Groq API keys configured. LLM calls will fail.")

    # ── Public ────────────────────────────────────────────────────────────────

    def call(self, messages: List[Dict[str, Any]]) -> str:
        """
        Send a chat completion request.

        Tries every model × every key until one succeeds.
        Raises the last exception if all combinations are exhausted.
        """
        if not self.api_keys:
            raise ValueError("No Groq API keys configured.")

        # Ensure the word 'json' appears in the prompt (required by json_object mode)
        messages = self._ensure_json_word(messages)

        last_exc: Exception | None = None

        for model in self.models:
            for offset in range(len(self.api_keys)):
                idx = (self._key_index + offset) % len(self.api_keys)
                key = self.api_keys[idx]
                try:
                    logger.info(
                        f"Groq call — model={model} key_idx={idx} (...{key[-6:]})"
                    )
                    client = Groq(api_key=key, timeout=self.timeout, max_retries=0)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    self._key_index = idx  # remember last successful key
                    return resp.choices[0].message.content

                except groq.RateLimitError as e:
                    last_exc = e
                    logger.warning(f"Rate limit (429) — model={model} key_idx={idx}. Rotating key.")
                except (groq.APIConnectionError, groq.APITimeoutError) as e:
                    last_exc = e
                    logger.warning(f"Connection/timeout — model={model} key_idx={idx}: {e}")
                except groq.APIStatusError as e:
                    last_exc = e
                    if e.status_code in (400, 404):
                        logger.warning(
                            f"Model '{model}' unsupported (HTTP {e.status_code}). "
                            "Skipping to next model."
                        )
                        break  # skip remaining keys for this model
                    elif e.status_code == 429:
                        logger.warning(f"Status 429 — model={model} key_idx={idx}. Rotating key.")
                    elif e.status_code in (401, 403):
                        logger.warning(f"Auth error (HTTP {e.status_code}) for key_idx={idx}.")
                    else:
                        logger.warning(f"API error HTTP {e.status_code} — {e.message}")
                except Exception as e:
                    last_exc = e
                    logger.error(f"Unexpected error — model={model}: {e}")
                    raise

            logger.warning(f"All keys exhausted for model '{model}'. Trying next model.")

        logger.error("All Groq models and API keys exhausted.")
        if last_exc:
            raise last_exc
        raise RuntimeError("Groq API call failed after trying all keys and models.")

    def call_parsed(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call Groq and return parsed dict. Raises ValueError on invalid JSON."""
        raw = self.call(messages)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Groq: {e}\nRaw: {raw[:500]}")
            raise ValueError(f"Groq returned invalid JSON: {e}") from e

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_json_word(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Groq's json_object mode requires the word 'json' in the prompt."""
        for msg in messages:
            if "json" in msg.get("content", "").lower():
                return messages
        # Append to last message
        modified = list(messages)
        last = modified[-1]
        modified[-1] = {
            "role": last["role"],
            "content": last["content"] + "\nReturn response in JSON format.",
        }
        return modified
