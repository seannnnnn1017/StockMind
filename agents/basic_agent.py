"""Multi-step calculator-only agent."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import List, Tuple

from llm_client import LLMClient, LLMMessage
from tools.calculator import CalculatorTool
from tools.helpers import LoopTool, MemoryTool

RESET = "\033[0m"
AGENT_COLOR = "\033[36m"
TOOL_COLOR = "\033[33m"


@dataclass(slots=True)
class PlanStep: 
    description: str
    expression: str
    store: str | None = None


@dataclass(slots=True)
class AgentResponse:
    task: str
    expression: str
    value: float
    reasoning: str | None = None
    steps: Tuple[PlanStep, ...] = ()


class PlanParseError(Exception):
    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class BasicAgent:
    """Plans step-by-step calculator executions using the LLM."""

    def __init__(self, llm: LLMClient, calculator: CalculatorTool | None = None) -> None:
        self.llm = llm
        self.calculator = calculator or CalculatorTool()
        self.loop = LoopTool()

    def solve(self, task: str) -> AgentResponse:
        print(f"{AGENT_COLOR}[agent] planning solution for task: {task}{RESET}", flush=True)
        steps, reasoning = self._plan_steps(task)
        for idx, step in enumerate(steps, start=1):
            print(f"{AGENT_COLOR}[agent] planned step {idx}: {step.description} -> {step.expression}{RESET}", flush=True)
        memory = MemoryTool()
        self._initialize_memory_from_task(task, memory)
        value, final_expr = self._execute_steps(steps, memory)
        return AgentResponse(task=task, expression=final_expr, value=value, reasoning=reasoning, steps=tuple(steps))

    def _plan_steps(self, task: str) -> Tuple[List[PlanStep], str | None]:
        system_base = (
            "You orchestrate a calculator tool. Each step must be a single arithmetic expression using numbers,"
            " + - * / ** %, parentheses, the literal constants pi/tau/e, and the math functions "
            f"{', '.join(sorted(CalculatorTool.allowed_functions.keys()))}."
            " Use the `store` field to name each result; reference previous values via {name}."
            " Do not emit memory.* calls, code, loops, tuples, or assignments—fully unroll the math."
            " Do NOT use math.pi. Use the literal constant 'pi'."
            " All constants must be written as pi, e, tau (no module prefixes)."
            " Respond with strict JSON: {\"reasoning\": str, \"steps\": [{\"description\": str, \"expression\": str, \"store\": str}]}."
        )

        last_error: str | None = None
        raw: str = ""

        for attempt in range(3):  # one initial try + one repair
            system = LLMMessage(
                role="system",
                content=(
                    system_base
                    + ("\n\nThe previous plan failed to parse because: " + last_error
                       + "\nRewrite the plan to strictly follow the tool constraints."
                       if last_error else "")
                ),
            )
            user = LLMMessage(role="user", content=f"Task: {task}\nReturn only JSON.")
            raw = self.llm.chat([system, user])
            print(
                f"{AGENT_COLOR}[agent] LLM plan response (attempt {attempt+1}):\n"
                f"{self._format_plan_text(raw)}{RESET}",
                flush=True,
            )

            try:
                return self._parse_plan(raw, task)
            except PlanParseError as e:
                last_error = str(e)
                print(f"\033[31m[agent] plan rejected: {last_error}\033[0m", flush=True)
                print(f"\033[35m[agent] system prompt was:\n{system.content}\033[0m", flush=True)
                print(
                    f"\033[35m[agent] user prompt was:\n{user.content}\033[0m",
                    flush=True,
                )
                continue

        # Final fallback if repair also fails
        fallback = PlanStep(description="Compute value", expression=self._fallback_expression(task))
        return [fallback], None

    def _parse_plan(self, raw: str, task: str) -> Tuple[List[PlanStep], str | None]:
        try:
            payload = raw[raw.find("{") : raw.rfind("}") + 1]
            data = json.loads(payload)
            steps: List[PlanStep] = []
            for idx, item in enumerate(data.get("steps", []), start=1):
                expression = str(item.get("expression", "")).strip()
                description = str(item.get("description", f"Step {idx}")).strip() or f"Step {idx}"
                store = (str(item.get("store")).strip() or None) if item.get("store") else None
                if not expression:
                    continue
                if "," in expression:
                    raise PlanParseError(f"Commas are not supported in expressions: {expression}", raw)
                if any(bad in expression for bad in ["=", "def", "lambda", "import", ";"]):
                    raise PlanParseError(
                        "Assignment, definitions, or statements are not allowed. "
                        f"Offending expression: {expression}", raw
                    )
                if "\n" in expression:
                    raise PlanParseError(
                        "Multi-line expressions are not supported; each step must be a single arithmetic expression.",
                        raw,
                    )
                placeholders = re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", expression)
                for placeholder in placeholders:
                    if placeholder in {"pi", "tau", "e"}:
                        raise PlanParseError(
                            f"Use literal constants (e.g., pi) rather than {{{placeholder}}}.", raw
                        )
                stripped = expression
                for placeholder in placeholders:
                    stripped = stripped.replace(f"{{{placeholder}}}", "")
                allowed_words = {"pi", "tau", "e"} | set(CalculatorTool.allowed_functions.keys())
                for word in allowed_words:
                    stripped = re.sub(rf"\b{word}\b", "", stripped)
                if re.search(r"[A-Za-z_]", stripped):
                    raise PlanParseError(
                        "Variables other than placeholders or supported function names are not allowed.", raw
                    )
                steps.append(PlanStep(description=description, expression=expression, store=store))
            if not steps:
                raise PlanParseError("No valid steps were produced.", raw)
            return steps, data.get("reasoning")
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Invalid JSON returned by LLM: {exc}", raw) from exc

    def _fallback_expression(self, task: str) -> str:
        lower = task.lower()
        if "pi" in lower:
            return str(math.pi)
        if " tau" in lower or lower.startswith("tau"):
            return str(math.tau)
        if " e" in lower or lower.startswith("e"):
            return str(math.e)
        digits = [c for c in task if c.isdigit() or c in ".+-*/()"]
        expr = "".join(digits).strip()
        return expr or "0"

    def _execute_steps(self, steps: List[PlanStep], memory: MemoryTool) -> Tuple[float, str]:
        last_value = 0.0
        last_expr = "0"
        for idx, step in enumerate(steps, start=1):
            substituted = self._substitute(step.expression, memory)
            normalized = self._normalize(substituted)
            print(f"{AGENT_COLOR}[agent] executing step {idx}: {step.description} -> {substituted}{RESET}", flush=True)
            print(f"{TOOL_COLOR}[tool] calculator.run({normalized}){RESET}", flush=True)
            try:
                value = self.calculator.run(normalized)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed on step {idx} ({step.description}): {exc}") from exc
            target = step.store or f"step{idx}"
            memory.set(target, value)
            last_value = value
            last_expr = normalized
        return last_value, last_expr

    def _substitute(self, expression: str, memory: MemoryTool) -> str:
        pattern = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            try:
                return str(memory.get(key))
            except KeyError as exc:
                raise ValueError(f"referenced value '{key}' is undefined") from exc

        return pattern.sub(replace, expression)

    def _normalize(self, expression: str) -> str:
        replacements = {"pi": str(math.pi), "tau": str(math.tau), "e": str(math.e)}
        pattern = re.compile(r"\b(pi|tau|e)\b")
        return pattern.sub(lambda m: replacements[m.group(0)], expression)

    def _initialize_memory_from_task(self, task: str, memory: MemoryTool) -> None:
        # Patterns like: x=2, x = 2, or 'x is 2'
        assignment_pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|is)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
        for name, value in assignment_pattern.findall(task):
            try:
                memory.set(name, float(value))
            except ValueError:
                continue

    def _format_plan_text(self, raw: str) -> str:
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                reasoning = data.get("reasoning", "")
                steps = data.get("steps", [])
                lines = []
                if reasoning:
                    lines.append(f"Reasoning:\n  {reasoning}")
                if steps:
                    lines.append("Steps:")
                    for idx, step in enumerate(steps, start=1):
                        desc = step.get("description", "")
                        expr = step.get("expression", "")
                        store = step.get("store", "")
                        lines.append(f"  {idx}. desc={desc} | expr={expr} | store={store}")
                return "\n".join(lines) or raw
            except json.JSONDecodeError:
                return raw
        return raw
