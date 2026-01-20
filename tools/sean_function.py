"""Utility tools: persistent memory and loop execution helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, MutableMapping


class MemoryTool:
    """Lightweight key-value store to share state across steps."""

    def __init__(self) -> None:
        self._store: MutableMapping[str, float] = {}

    def get(self, name: str, default: float = 0.0) -> float:
        """Return the stored value or a default."""
        return self._store.get(name, default)

    def set(self, name: str, value: float) -> None:
        """Persist a numeric value under the provided name."""
        self._store[name] = float(value)


@dataclass
class LoopTool:
    """Runs a sequence of step callables repeatedly."""

    steps: List[Callable[[int], Any]] = field(default_factory=list)

    def repeat(self, n: int, steps: Iterable[Callable[[int], Any]] | None = None) -> List[Any]:
        """Execute the given steps n times, collecting their outputs."""
        sequence = list(steps) if steps is not None else self.steps
        results: List[Any] = []
        for iteration in range(n):
            for step in sequence:
                results.append(step(iteration))
        return results
