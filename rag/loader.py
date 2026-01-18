"""Utilities for loading raw PDF transcripts."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader


def load_pdf(path: Path) -> List[str]:
    """Return one string per page for downstream chunking."""
    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages
