"""Lightweight evaluator for answers and citations."""
from __future__ import annotations

from typing import Dict, List


def score_answer(prediction: str, references: List[str]) -> Dict[str, float]:
    """Compute toy accuracy and citation coverage metrics."""
    accuracy = 1.0 if any(ref.lower() in prediction.lower() for ref in references) else 0.0
    citation = prediction.count("[")
    return {"accuracy": accuracy, "citations": float(citation)}
