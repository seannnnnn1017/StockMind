"""Critic agent that evaluates drafts for accuracy and clarity."""
from __future__ import annotations

from typing import Tuple


class CriticAgent:
    """Lightweight QA loop to keep hallucinations in check."""

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def review(self, draft: str) -> Tuple[str, bool]:
        """Return a revised draft and whether it passed QA."""
        if not draft.strip():
            return ("", False)
        critique = self._llm.generate(
            "Assess the following analysis for factual issues and fix them:\n" + draft
        )
        approved = "APPROVED" in critique.upper()
        return critique, approved
