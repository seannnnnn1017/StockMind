"""Performs financial reasoning using reader output and context memories."""
from __future__ import annotations

from typing import Iterable, List

from .reader_agent import SummaryPoint


class AnalystAgent:
    """Turns narrative summaries into investment-ready insights."""

    def __init__(self, llm_client, templates: Iterable[str]) -> None:
        self._llm = llm_client
        self._templates = list(templates)

    def draft_analysis(self, points: List[SummaryPoint]) -> str:
        """Apply prompts to produce a coherent earnings assessment."""
        prompt_chunks = [template.format(points=points) for template in self._templates]
        prompt = "\n\n".join(prompt_chunks)
        return self._llm.generate(prompt)
