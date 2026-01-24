"""Runtime configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeSettings:
    """Minimal settings bundle for the local agent stack."""

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen/qwen3-8b"
    temperature: float = 0.1
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
            model=os.getenv("LLM_MODEL", "qwen/qwen3-8b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60")),
        )
