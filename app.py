"""AlphaMind CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from agents.analyst_agent import AnalystAgent
from agents.critic_agent import CriticAgent
from agents.memory_agent import MemoryAgent
from agents.reader_agent import ReaderAgent
from rag import chunker, embedder, loader


class EchoLLM:
    """Minimal LLM stub so the skeleton can run offline."""

    def generate(self, prompt: str) -> str:
        return f"[LLM OUTPUT]\n{prompt[:200]}..."


def run_pipeline(pdf_path: Path) -> None:
    llm = EchoLLM()
    pages = loader.load_pdf(pdf_path)
    chunks = chunker.chunk_text(pages)
    embedder_model = embedder.SimpleEmbedder()
    vectors = embedder_model.encode(chunks)

    reader = ReaderAgent(llm)
    analyst = AnalystAgent(llm, templates=["{points}"])
    critic = CriticAgent(llm)
    memory = MemoryAgent(Path("data/metadata"))

    summary = reader.summarize("\n".join(pages))
    analysis = analyst.draft_analysis(summary)
    critique, _ = critic.review(analysis)
    memory.store("demo", "Q1", [critique])

    print("Chunks:", len(chunks))
    print("Vectors:", len(vectors))
    print("Critique:\n", critique)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaMind earnings pipeline")
    parser.add_argument("pdf", type=Path, help="Path to the earnings call PDF")
    args = parser.parse_args()
    run_pipeline(args.pdf)


if __name__ == "__main__":
    main()
