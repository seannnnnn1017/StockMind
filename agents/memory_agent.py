"""Memory agent responsible for long-term storage of insights."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


class MemoryAgent:
    """Writes learnings to disk for retrieval in future quarters."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def store(self, company: str, quarter: str, notes: Iterable[str]) -> Path:
        """Persist the agent notes grouped by company and quarter."""
        target = self._storage_dir / f"{company}_{quarter}.md"
        with target.open("a", encoding="utf-8") as handle:
            for note in notes:
                handle.write(f"- {note}\n")
        return target
