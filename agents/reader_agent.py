"""Reader agent that ingests transcripts and generates structured summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SummaryPoint:
    """Single insight extracted from an earnings call."""

    section: str
    takeaway: str
    evidence: str


class ReaderAgent:
    """Pulls cleaned text, calls the LLM, and normalizes the summary."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def summarize(self, transcript: str) -> List[SummaryPoint]:
        """Return structured points so downstream agents can reason easily."""
        if not transcript.strip():
            return []
        prompt = "Summarize the following earnings call in 3 bullet points:" + "\n" + transcript
        response = self._llm.generate(prompt)
        return self._parse_response(response)

    def _parse_response(self, text: str) -> List[SummaryPoint]:
        points: List[SummaryPoint] = []
        for line in text.splitlines():
            line = line.strip("- •")
            if not line:
                continue
            points.append(
                SummaryPoint(section="general", takeaway=line, evidence="model")
            )
        return points
