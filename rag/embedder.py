"""Transform chunks into dense embeddings."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


class SimpleEmbedder:
    """Toy embedder that averages character codes to mimic vectors."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def encode(self, chunks: Iterable[str]) -> List[np.ndarray]:
        vectors: List[np.ndarray] = []
        for chunk in chunks:
            rng = np.random.default_rng(abs(hash(chunk)) % (2**32))
            vectors.append(rng.standard_normal(self._dim))
        return vectors
