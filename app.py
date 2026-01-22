"""Minimal calculator agent wired to LM Studio."""
from __future__ import annotations

import argparse
import sys

from agents.basic_agent import AgentResponse, BasicAgent
from config.runtime import RuntimeSettings
from llm_client import LLMClient

RESET = "\033[0m"
INFO = "\033[35m"
RESULT = "\033[32m"

if not sys.stdout.isatty():
    RESET = INFO = RESULT = ""


def build_llm(use_mock: bool) -> LLMClient:
    settings = RuntimeSettings.from_env()
    return LLMClient(settings=settings, use_mock=use_mock)


def run_task(prompt: str, use_mock: bool = False) -> None:
    llm = build_llm(use_mock)
    agent = BasicAgent(llm)
    solution: AgentResponse = agent.solve(prompt)

    print(f"{INFO}User task:{RESET} {solution.task}", flush=True)
    if solution.steps:
        print(f"{INFO}Plan steps:{RESET}", flush=True)
        for idx, step in enumerate(solution.steps, start=1):
            if hasattr(step, "expression"):
                store = getattr(step, "store", None) or "-"
                desc = getattr(step, "description", f"Step {idx}")
                expr = getattr(step, "expression", "")
                print(f"  {idx}. {desc} -> {expr} [store={store}]", flush=True)
            else:
                print(f"  {idx}. {step}", flush=True)
    if solution.expression:
        print(f"{INFO}Expression:{RESET} {solution.expression}", flush=True)
    if solution.value is not None:
        print(f"{RESULT}Result:{RESET} {solution.value}", flush=True)
    if solution.answer and solution.value is None:
        print(f"{RESULT}Answer:{RESET} {solution.answer}", flush=True)
    if solution.outputs:
        print(f"{RESULT}Outputs:{RESET}", flush=True)
        for name, value in solution.outputs:
            print(f"  - {name}: {value}", flush=True)
    if solution.reasoning:
        print(f"{INFO}Reasoning:{RESET} {solution.reasoning}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent calculator pipeline")
    parser.add_argument("prompt", help="User query to solve")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM responses (offline testing)")
    args = parser.parse_args()
    run_task(args.prompt, use_mock=args.mock)


if __name__ == "__main__":
    main()
