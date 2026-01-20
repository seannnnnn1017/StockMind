"""Utility tools for storing intermediate values and repeating steps."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Dict, List, MutableMapping


class MemoryTool:
    """
    Tool: memory

    Purpose:
      Persistent key-value storage for intermediate scalar values used across steps.

    State:
      - Keys: string identifiers (SSA-style; once set, should be treated as immutable by the agent).
      - Values: floats (all inputs are coerced to float).

    API:
      - set(name: str, value: float) -> None
        Stores a numeric value under `name`.
      - get(name: str, default: float | None = None) -> float
        Retrieves the stored value. If missing and `default` is None, raises KeyError.
      - snapshot() -> Dict[str, float]
        Returns a copy of the current memory.

    Semantics:
      - Deterministic, in-process.
      - No side effects beyond updating internal state.
      - Values are numeric only.

    Agent Usage Rules:
      - Every computed result intended for reuse must be stored via `set`.
      - Refer to stored values by placeholder `{name}` in subsequent tool expressions (do not call get() inside expressions).
      - A name must be defined before it is referenced.
      - Treat names as immutable (single-assignment discipline).
    """

    def __init__(self) -> None:
        self._values: MutableMapping[str, float] = {}

    def set(self, name: str, value: float) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Memory key must be a non-empty string.")
        v = float(value)
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("Memory value must be a finite number.")
        self._values[name] = v

    def get(self, name: str, default: float | None = None) -> float:
        if name not in self._values:
            if default is None:
                raise KeyError(f"{name} not found in memory")
            return float(default)
        return float(self._values[name])

    def snapshot(self) -> Dict[str, float]:
        return dict(self._values)


class LoopTool:
    """
    Tool: repeat

    Purpose:
      Deterministically unroll iteration by executing a fixed sequence of steps
      a given number of times. This provides controlled repetition without
      implicit loops in the agent plan.

    API:
      - repeat(count: int, steps: Iterable[Callable[[int], Any]]) -> List[Any]

    Semantics:
      - For idx in [0, count):
          For each step in `steps`:
            call step(idx) and append its return value to the results list.
      - Order is preserved.
      - Deterministic given deterministic callables.

    Agent Usage Rules:
      - `count` must be a non-negative integer known at plan time.
      - `steps` must be a finite, ordered collection of callables.
      - No hidden state: all state changes should occur via explicit tools (e.g., memory.set).
      - Prefer unrolling via `repeat` rather than using language-level loops in plans.
    """

    def repeat(self, count: int, steps: Iterable[Callable[[int], Any]]) -> List[Any]:
        if not isinstance(count, int) or count < 0:
            raise ValueError("repeat(count, ...) requires a non-negative integer count.")
        results: List[Any] = []
        for idx in range(count):
            for step in steps:
                results.append(step(idx))
        return results
