"""Simple text chunker for splitting pages into overlapping windows."""
from __future__ import annotations

from typing import Iterable, List


def chunk_text(pages: Iterable[str], size: int = 800, overlap: int = 100) -> List[str]:
    """Chunk each page and keep overlaps for better retrieval."""
    chunks: List[str] = []
    for page in pages:
        start = 0
        while start < len(page):
            end = min(len(page), start + size)
            chunks.append(page[start:end])
            start += size - overlap
            if start < 0:
                break
    return chunks
