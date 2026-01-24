"""Extract fields from a stock quote."""
from __future__ import annotations

from typing import Any, Dict


class QuoteFieldTool:
    """
    Tool: quote_field

    Purpose:
      Extract a numeric field from a stock quote dict.

    Input:
      quote: dict
      field: string (open/high/low/close/volume)
    """

    allowed_fields = {"open", "high", "low", "close", "volume"}

    def run(self, quote: Dict[str, Any], field: str) -> float:
        field = field.strip().lower()
        if field not in self.allowed_fields:
            raise ValueError(f"Unsupported quote field: {field}")
        if field not in quote:
            raise ValueError(f"Quote missing field: {field}")
        return float(quote[field])
