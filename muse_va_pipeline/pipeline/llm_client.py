"""Configurable LLM client adapters for the MUSE-VA text pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 120

    @classmethod
    def from_env(
        cls,
        provider: str = "mock",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "LLMConfig":
        provider = provider.lower()
        if provider == "gemini":
            return cls(
                provider=provider,
                model=model or os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro"),
                api_key=api_key or os.getenv("GEMINI_API_KEY"),
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            )
        if provider in {"openai", "openai-compatible"}:
            return cls(
                provider="openai-compatible",
                model=model or os.getenv("OPENAI_COMPATIBLE_MODEL"),
                api_key=api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY"),
                base_url=base_url
                or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1"),
                timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            )
        return cls(provider=provider, model=model, api_key=api_key, base_url=base_url)


def call_gemini(prompt: str, config: LLMConfig) -> str:
    if not config.api_key:
        raise ValueError("GEMINI_API_KEY is required for provider='gemini'.")
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for provider='gemini'."
        ) from exc

    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(config.model or "models/gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()


def call_openai_compatible(prompt: str, config: LLMConfig) -> str:
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests is required for provider='openai-compatible'."
        ) from exc

    if not config.api_key:
        raise ValueError(
            "OPENAI_COMPATIBLE_API_KEY is required for provider='openai-compatible'."
        )
    if not config.model:
        raise ValueError(
            "OPENAI_COMPATIBLE_MODEL or --llm-model is required for provider='openai-compatible'."
        )

    base_url = (config.base_url or "").rstrip("/")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        timeout=config.timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload["choices"][0]["message"]["content"].strip()


def call_mock(prompt: str) -> str:
    """Return deterministic valid JSON for local smoke tests."""
    if "consistency" in prompt:
        return json.dumps(
            {
                "consistency": {
                    "result": True,
                    "reason": "The affective label, music description, and visual imagery are mutually aligned.",
                }
            },
            ensure_ascii=False,
        )
    if (
        "visual_imagery" in prompt
        and "visual_tags" in prompt
        and "visual_caption" in prompt
    ):
        return json.dumps(
            {
                "visual_imagery": (
                    "A quiet open landscape at dusk, with soft light, long shadows, "
                    "and a calm sense of motion matching the instrumental texture."
                ),
                "visual_tags": "dusk, open landscape, soft light, calm motion, cinematic",
                "visual_caption": (
                    "A cinematic dusk landscape with soft light and a calm emotional tone."
                ),
            },
            ensure_ascii=False,
        )
    if "caption_full" in prompt and "caption_tags" in prompt:
        return json.dumps(
            {
                "caption_full": (
                    "An instrumental ambient piece led by gentle piano, supported "
                    "by warm synthesizer pads and a subtle drum kit pulse, with a "
                    "balanced emotional atmosphere shaped by the target valence and arousal."
                ),
                "caption_tags": "ambient, instrumental, piano, synthesizer, subtle drums",
            },
            ensure_ascii=False,
        )
    if "genre" in prompt and "lead_instruments" in prompt and "composition_notes" in prompt:
        return json.dumps(
            {
                "genre": "Ambient",
                "lead_instruments": ["Piano"],
                "supporting_instruments": ["Synthesizer", "Drum kit"],
                "tempo": 96,
                "key": "C Major",
                "composition_notes": (
                    "Use a clear melodic piano line, soft synthesizer harmony, "
                    "and restrained percussion to express the target affect."
                ),
            },
            ensure_ascii=False,
        )
    raise ValueError("Mock LLM could not infer the requested stage from the prompt.")


def call_llm(
    model_name: str | None = None,
    prompt: str | None = None,
    *,
    config: LLMConfig | None = None,
    **_: Any,
) -> str:
    """Compatibility dispatcher used by stage modules.

    Existing stage functions pass ``model_name``. New code may pass a full
    ``LLMConfig`` for provider/model/base-url control.
    """
    if prompt is None:
        raise ValueError("prompt is required")
    cfg = config or LLMConfig.from_env(provider=model_name or "mock")
    if cfg.provider == "mock":
        return call_mock(prompt)
    if cfg.provider == "gemini":
        return call_gemini(prompt, cfg)
    if cfg.provider == "openai-compatible":
        return call_openai_compatible(prompt, cfg)
    raise ValueError(
        f"Unknown LLM provider: {cfg.provider}. Use mock, gemini, or openai-compatible."
    )
