"""Runtime configuration helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeSettings:
    """Minimal settings bundle for the local agent stack."""

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "wen/Qwen2.5-7B-Instruct-GGUF"
    temperature: float = 0.1
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
            model=os.getenv("LLM_MODEL", "wen/Qwen2.5-7B-Instruct-GGUF"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
        )
