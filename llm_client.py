"""HTTP client for LM Studio (OpenAI compatible) plus a mock fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

import json
import os

import requests

from config.runtime import RuntimeSettings


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str

    def as_dict(self) -> Mapping[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    def __init__(
        self,
        settings: RuntimeSettings,
        api_key: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self.settings = settings
        self.api_key = api_key or os.getenv("LM_STUDIO_API_KEY")
        self.use_mock = use_mock

    def chat(self, messages: Iterable[LLMMessage], temperature: Optional[float] = None) -> str:
        if self.use_mock:
            return self._mock_response(messages)

        payload = {
            "model": self.settings.model,
            "messages": [msg.as_dict() for msg in messages],
            "temperature": temperature if temperature is not None else self.settings.temperature,
        }
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(url, headers=headers, timeout=self.settings.timeout, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response: {data}") from exc

    def _mock_response(self, messages: Iterable[LLMMessage]) -> str:
        last_user = next((m.content for m in reversed(list(messages)) if m.role == "user"), "")
        return f"[MOCK RESPONSE]\n{last_user.strip()}"
