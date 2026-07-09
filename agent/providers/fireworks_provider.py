"""Fireworks AI provider (OpenAI-compatible endpoint)."""

from __future__ import annotations

from .openai_provider import OpenAIProvider

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksProvider(OpenAIProvider):
    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str) -> None:
        super().__init__(model, temperature, max_tokens, api_key, FIREWORKS_BASE_URL)
