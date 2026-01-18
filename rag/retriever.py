"""Naive similarity search over embedded chunks."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class Retriever:
    """Keeps embeddings in memory and performs cosine similarity."""

    def __init__(self, embeddings: List[np.ndarray], chunks: List[str]) -> None:
        self._embeddings = [self._normalize(vec) for vec in embeddings]
        self._chunks = chunks

    def top_k(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        query = self._normalize(query_vec)
        for chunk, vec in zip(self._chunks, self._embeddings):
            score = float(np.dot(query, vec))
            scores.append((chunk, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm
