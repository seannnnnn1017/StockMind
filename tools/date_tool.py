"""Date lookup tool."""
from __future__ import annotations

from datetime import datetime


class DateTool:
    """
    Tool: date

    Purpose:
      Return the current local date.

    Input:
      None
    """

    def run(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")
